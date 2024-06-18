import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import classification
import wandb
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from huggingface_hub import PyTorchModelHubMixin
from lion_pytorch import Lion

import json

from messis.prithvi import TemporalViTEncoder, ConvTransformerTokensToEmbeddingNeck


def safe_shape(x):
    if isinstance(x, tuple):
        # loop through tuple
        shape_info = '(tuple) : '
        for i in x:
            shape_info += str(i.shape) + ', '
        return shape_info
    if isinstance(x, list):
        # loop through list
        shape_info = '(list) : '
        for i in x:
            shape_info += str(i.shape) + ', '
        return shape_info
    return x.shape

class ConvModule(nn.Module):
    """
    A simple convolutional module including Conv, BatchNorm, and ReLU layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class HierarchicalFCNHead(nn.Module):
    """
    Hierarchical FCN Head for semantic segmentation.
    """
    def __init__(self, in_channels, out_channels, num_classes, num_convs=2, kernel_size=3, dilation=1, dropout_p=0.1, debug=False):
        super(HierarchicalFCNHead, self).__init__()

        self.debug = debug
        
        self.convs = nn.Sequential(*[
            ConvModule(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size,
                padding=dilation * (kernel_size // 2),
                dilation=dilation
            ) for i in range(num_convs)
        ])
        
        self.conv_seg = nn.Conv2d(out_channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        if self.debug:
            print('HierarchicalFCNHead forward INP: ', safe_shape(x))
        x = self.convs(x)
        features = self.dropout(x)
        output = self.conv_seg(features)
        if self.debug:
            print('HierarchicalFCNHead forward features OUT: ', safe_shape(features))
            print('HierarchicalFCNHead forward output OUT: ', safe_shape(output))
        return output, features

class LabelRefinementHead(nn.Module):
    """
    Similar to the label refinement module introduced in the ZueriCrop paper, this module refines the predictions for tier 3.
    It takes the raw predictions from head 1, head 2 and head 3 and refines them to produce the final prediction for tier 3.
    According to ZueriCrop, this helps with making the predictions more consistent across the different tiers.
    """
    def __init__(self, input_channels, num_classes):
        super(LabelRefinementHead, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            # 1x1 Convolutional layer
            nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 3x3 Convolutional layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # Skip connection (implemented in forward method)
            
            # Another 3x3 Convolutional layer
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 1x1 Convolutional layer to adjust the number of output channels to num_classes
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )
        
    def forward(self, x):
        # Apply initial conv layer
        y = self.cnn_layers[0:3](x)

        # Save for skip connection
        y_skip = y

        # Apply the next two conv layers
        y = self.cnn_layers[3:9](y)

        # Skip connection (element-wise addition)
        y = y + y_skip

        # Apply the last conv layer
        y = self.cnn_layers[9:](y)
        return y

class HierarchicalClassifier(nn.Module):
    def __init__(
            self, 
            heads_spec,
            dropout_p=0.1,
            img_size=256, 
            patch_size=16, 
            num_frames=3, 
            bands=[0, 1, 2, 3, 4, 5], 
            backbone_weights_path=None, 
            freeze_backbone=True, 
            debug=False
        ):
        super(HierarchicalClassifier, self).__init__()

        self.embed_dim = 768
        self.num_frames = num_frames
        self.output_embed_dim = self.embed_dim * self.num_frames
        self.hp, self.wp = img_size // patch_size, img_size // patch_size
        self.head_channels = 256 # TODO: We should research what makes sense here (same channels, gradual decrease from 1024, ...)
        self.heads_spec = heads_spec
        self.dropout_p = dropout_p
        self.debug = debug

        if self.debug:
            print('hp and wp: ', self.hp, self.wp)

        self.prithvi = TemporalViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=self.num_frames,
            tubelet_size=1,
            in_chans=len(bands),
            embed_dim=self.embed_dim,
            depth=12,
            num_heads=8,
            mlp_ratio=4.0,
            norm_pix_loss=False,
            pretrained=backbone_weights_path,
            debug=self.debug
        )

        # (Un)freeze the backbone
        for param in self.prithvi.parameters():
            param.requires_grad = not freeze_backbone

        # Neck to transform the token-based output of the transformer into a spatial feature map
        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=self.embed_dim * self.num_frames,
            output_embed_dim=self.output_embed_dim,
            drop_cls_token=True,
            Hp=self.hp,
            Wp=self.wp,
        )

        # Initialize heads and loss weights based on tiers
        self.heads = nn.ModuleDict()
        self.loss_weights = {}
        self.total_classes = 0

        # Build HierarchicalFCNHeads
        head_count = 0
        for head_name, head_info in self.heads_spec.items():
            head_type = head_info['type']
            num_classes = head_info['num_classes_to_predict']
            loss_weight = head_info['loss_weight']

            if head_type == 'HierarchicalFCNHead':
                num_classes = head_info['num_classes_to_predict']
                loss_weight = head_info['loss_weight']
                self.total_classes += num_classes

                self.heads[head_name] = HierarchicalFCNHead(
                    in_channels=self.output_embed_dim if head_count == 0 else self.head_channels,
                    out_channels=self.head_channels,
                    num_classes=num_classes,
                    num_convs=1,
                    dropout_p=self.dropout_p,
                    debug=self.debug
                )
                self.loss_weights[head_name] = loss_weight

            # NOTE: LabelRefinementHead must be the last in the dict, otherwise the total_classes will be incorrect
            if head_type == 'LabelRefinementHead':
                self.refinement_head = LabelRefinementHead(input_channels=self.total_classes, num_classes=num_classes)
                self.refinement_head_name = head_name
                self.loss_weights[head_name] = loss_weight

            head_count += 1

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        # Extract features from the base model
        features = self.prithvi(x)

        if self.debug:
            print(f"Features shape after base model: {safe_shape(features)}")

        # Process through the neck
        features = self.neck(features)

        if self.debug:
            print(f"Features shape after neck: {safe_shape(features)}")

        # Remove from tuple
        features = features[0]
        if self.debug:
            print(f"Features shape after removing tuple: {safe_shape(features)}")

        # Process through the heads
        outputs = {}
        for tier_name, head in self.heads.items():
            output, features = head(features)
            outputs[tier_name] = output

            if self.debug:
                print(f"Features shape after {tier_name} head: {safe_shape(features)}")
                print(f"Output shape after {tier_name} head: {safe_shape(output)}")

        # Process through the classification refinement head
        output_concatenated = torch.cat(list(outputs.values()), dim=1)
        output_refinement_head = self.refinement_head(output_concatenated)
        outputs[self.refinement_head_name] = output_refinement_head

        return outputs

    def calculate_loss(self, outputs, targets):
        total_loss = 0
        loss_per_head = {}
        for head_name, output in outputs.items():
            if self.debug:
                print(f"Target index for {head_name}: {self.heads_spec[head_name]['target_idx']}")
            target = targets[self.heads_spec[head_name]['target_idx']]
            loss = self.loss_func(output, target)
            loss_per_head[f'loss_{head_name}'] = loss
            total_loss += loss * self.loss_weights[head_name]
        
        return total_loss, loss_per_head

class Messis(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = HierarchicalClassifier(
            heads_spec=hparams['heads_spec'],
            dropout_p=hparams.get('dropout_p'),
            img_size=hparams.get('img_size'),
            patch_size=hparams.get('patch_size'),
            num_frames=hparams.get('num_frames'),
            bands=hparams.get('bands'),
            backbone_weights_path=hparams.get('backbone_weights_path'),
            freeze_backbone=hparams['freeze_backbone'],
            debug=hparams.get('debug')
        )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "test")
        
    def configure_optimizers(self):
        # select case on optimizer
        match self.hparams.get('optimizer', 'Adam'):
            case 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get('lr', 1e-3))
            case 'AdamW':
                optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.get('lr', 1e-3), weight_decay=self.hparams.get('optimizer_weight_decay', 0.01))
            case 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.get('lr', 1e-3), momentum=self.hparams.get('optimizer_momentum', 0.9))
            case 'Lion':
                # https://github.com/lucidrains/lion-pytorch | Typically lr 3-10 times lower than Adam and weight_decay 3-10 times higher
                optimizer = Lion(self.parameters(), lr=self.hparams.get('lr', 1e-3))
            case _:
                raise ValueError(f"Optimizer {self.hparams.get('optimizer')} not supported")
        return optimizer

    def __step(self, batch, batch_idx, stage):
        inputs, targets = batch
        targets = torch.stack(targets[0])
        outputs = self(inputs)
        loss, loss_per_head = self.model.calculate_loss(outputs, targets)
        loss_proportions = { f'loss_{head}_proportion': round(loss_per_head[head].item() / loss.item(), 2) for head in loss_per_head}
        loss_detail_dict = {**loss_per_head, **loss_proportions}

        if self.hparams.get('debug'):
            print(f"Step Inputs shape: {safe_shape(inputs)}")
            print(f"Step Targets shape: {safe_shape(targets)}")
            print(f"Step Outputs shape: {safe_shape(outputs)}")

        # NOTE: All metrics other than loss are tracked by callbacks (LogMessisMetrics)
        self.log_dict({f'{stage}_loss': loss, **loss_detail_dict}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'outputs': outputs}
        
class LogConfusionMatrix(pl.Callback):
    def __init__(self, hparams, dataset_info_file, debug=False):
        super().__init__()

        assert hparams.get('tiers') is not None, "Tiers must be defined in the hparams"

        self.tiers_dict = hparams.get('tiers')
        self.tiers = ['tier1', 'tier2', 'tier3']
        self.phases = ['train', 'val', 'test']
        self.modes = ['pixelwise', 'majority']
        self.debug = debug
        
        with open(dataset_info_file, 'r') as f:
            self.dataset_info = json.load(f)

        # Initialize confusion matrices
        self.metrics_to_compute = ['confusion_matrix']
        self.metrics = {phase: {tier: {mode: self.__init_metrics(tier, phase) for mode in self.modes} for tier in self.tiers} for phase in self.phases}

    def __init_metrics(self, tier, phase):
        num_classes = self.tiers_dict[tier]['num_classes']
        confusion_matrix = classification.MulticlassConfusionMatrix(num_classes=num_classes)

        return {
            'confusion_matrix': confusion_matrix
        }

    def setup(self, trainer, pl_module, stage=None):
        # Move all metrics to the correct device at the start of the training/validation
        device = pl_module.device
        for phase_metrics in self.metrics.values():
            for tier_metrics in phase_metrics.values():
                for mode_metrics in tier_metrics.values():
                    for metric in self.metrics_to_compute:
                        mode_metrics[metric].to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__update_confusion_matrices(trainer, pl_module, outputs, batch, batch_idx, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__update_confusion_matrices(trainer, pl_module, outputs, batch, batch_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__update_confusion_matrices(trainer, pl_module, outputs, batch, batch_idx, 'test')

    def __update_confusion_matrices(self, trainer, pl_module, outputs, batch, batch_idx, phase):
        if trainer.sanity_checking:
            return

        targets = torch.stack(batch[1][0]) # (tiers, batch, H, W)
        outputs = outputs['outputs'][3] # (batch, C, H, W)
        field_ids = batch[1][1].permute(1, 0, 2, 3)[0]

        pixelwise_outputs, majority_outputs = LogConfusionMatrix.get_pixelwise_and_majority_outputs(outputs, field_ids, self.dataset_info)        
        
        for preds, mode in zip([pixelwise_outputs, majority_outputs], self.modes):
            # Update all metrics
            assert len(preds) == len(targets), f"Number of predictions and targets do not match: {len(preds)} vs {len(targets)}"
            assert len(preds) == len(self.tiers), f"Number of predictions and tiers do not match: {len(preds)} vs {len(self.tiers)}"

            for pred, target, tier in zip(preds, targets, self.tiers):
                if self.debug:
                    print(f"Updating confusion matrix for {phase} {tier} {mode}")
                metrics = self.metrics[phase][tier][mode]
                # flatten and remove background class if the mode is majority (such that the background class is not included in the confusion matrix)
                if mode == 'majority':
                    pred = pred[target != 0]
                    target = target[target != 0]
                metrics['confusion_matrix'].update(pred, target)


    @staticmethod
    def get_pixelwise_and_majority_outputs(outputs, field_ids, dataset_info):
        """
        Get the pixelwise and majority predictions from the model outputs.
        The pixelwise tier 1 and 2 are derived from the tier3_refined predictions. 
        The majority tier 3 is first derived from the tier3_refined predictions. And then the majority tier 1 and 2 are derived from the majority tier 3 predictions.

        Also sets the background to 0 for all field majority predictions (regardless of what the model predicts for the background class).
        As this is a classification task and not a segmentation task and the field boundaries are known beforehand and not of any interest.

        Args:
            outputs (torch.Tensor(batch, C, H, W)): The probability outputs from the model for tier3_refined.
            field_ids (torch.Tensor(batch, H, W)): The field IDs for each prediction.
            dataset_info (dict): The dataset information.

        Returns:
            torch.Tensor(tiers, batch, H, W): The pixelwise predictions.
            torch.Tensor(tiers, batch, H, W): The majority predictions.
        """
        
        pixelwise_tier3 = torch.softmax(outputs, dim=1).argmax(dim=1) # (batch, H, W)
        majority_tier3 = LogConfusionMatrix.get_field_majority_preds(outputs, field_ids)

        tier3_to_tier1, tier3_to_tier2 = dataset_info['tier3_to_tier1'], dataset_info['tier3_to_tier2']

        pixelwise_tier1, pixelwise_tier2 = torch.zeros_like(pixelwise_tier3), torch.zeros_like(pixelwise_tier3)
        majority_tier1, majority_tier2 = torch.zeros_like(majority_tier3), torch.zeros_like(majority_tier3)

        # map tier 3 to tier 2 and tier 1
        for i, (class_index_tier1, class_index_tier2) in enumerate(zip(tier3_to_tier1, tier3_to_tier2)):
            pixelwise_tier1[pixelwise_tier3 == i] = class_index_tier1
            pixelwise_tier2[pixelwise_tier3 == i] = class_index_tier2
            majority_tier1[majority_tier3 == i] = class_index_tier1
            majority_tier2[majority_tier3 == i] = class_index_tier2
        
        pixelwise_outputs = torch.stack((pixelwise_tier1, pixelwise_tier2, pixelwise_tier3))
        majority_outputs = torch.stack((majority_tier1, majority_tier2, majority_tier3))

        # Ensure these are tensors
        assert isinstance(pixelwise_outputs, torch.Tensor), "pixelwise_outputs is not a tensor"
        assert isinstance(majority_outputs, torch.Tensor), "majority_outputs is not a tensor"

        return pixelwise_outputs, majority_outputs


    @staticmethod
    def get_field_majority_preds(output, field_ids):
        """
        Get the majority prediction for each field in the batch. The majority excludes the background class.

        Args:
            output (torch.Tensor(batch, C, H, W)): The probability outputs from the model (tier3_refined)
            field_ids (torch.Tensor(batch, H, W)): The field IDs for each prediction.

        Returns:
            torch.Tensor(batch, H, W): The majority predictions.
        """
        # remove the background class
        pixelwise = torch.softmax(output[:, 1:, :, :], dim=1).argmax(dim=1) + 1  # (batch, H, W)
        majority_preds = torch.zeros_like(pixelwise)
        for batch in range(len(pixelwise)):
            field_ids_batch = field_ids[batch]
            for field_id in np.unique(field_ids_batch.cpu().numpy()):
                if field_id == 0:
                    continue
                field_mask = field_ids_batch == field_id
                flattened_pred = pixelwise[batch][field_mask].view(-1)  # Flatten the prediction
                flattened_pred = flattened_pred[flattened_pred != 0]  # Exclude background class
                if len(flattened_pred) == 0:
                    continue
                mode_pred, _ = torch.mode(flattened_pred) # Compute mode prediction
                majority_preds[batch][field_mask] = mode_pred.item()
        return majority_preds

    def on_train_epoch_end(self, trainer, pl_module):
        # Log and then reset the confusion matrices after training epoch
        self.__log_and_reset_confusion_matrices(trainer, pl_module, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log and then reset the confusion matrices after validation epoch
        self.__log_and_reset_confusion_matrices(trainer, pl_module, 'val')

    def on_test_epoch_end(self, trainer, pl_module):
        # Log and then reset the confusion matrices after test epoch
        self.__log_and_reset_confusion_matrices(trainer, pl_module, 'test')

    def __log_and_reset_confusion_matrices(self, trainer, pl_module, phase):
        if trainer.sanity_checking:
            return

        for tier in self.tiers:
            for mode in self.modes:
                metrics = self.metrics[phase][tier][mode]
                confusion_matrix = metrics['confusion_matrix']
                if self.debug:
                    print(f"Logging and resetting confusion matrix for {phase} {tier} Update count: {confusion_matrix._update_count}")
                matrix = confusion_matrix.compute()  # columns are predictions and rows are targets

                # Calculate percentages
                matrix = matrix.float()
                row_sums = matrix.sum(dim=1, keepdim=True)
                matrix_percent = matrix / row_sums

                # Ensure percentages sum to 1 for each row or handle NaNs
                row_sum_check = matrix_percent.sum(dim=1)
                valid_rows = ~torch.isnan(row_sum_check)
                if valid_rows.any():
                    assert torch.allclose(row_sum_check[valid_rows], torch.ones_like(row_sum_check[valid_rows]), atol=1e-2), "Percentages do not sum to 1 for some valid rows"
                    
                # Check for zero rows
                zero_rows = (row_sums == 0).squeeze()

                fig, ax = plt.subplots(figsize=(matrix.size(0), matrix.size(0)), dpi=100)

                ax.matshow(matrix_percent.cpu().numpy(), cmap='viridis')

                ax.xaxis.set_major_locator(ticker.FixedLocator(range(matrix.size(1)+1)))
                ax.yaxis.set_major_locator(ticker.FixedLocator(range(matrix.size(0)+1)))

                clean_tier = tier.split('_')[0] if '_refined' in tier else tier
                ax.set_xticklabels(self.dataset_info[clean_tier] + [''], rotation=45)
                ax.set_yticklabels(self.dataset_info[clean_tier] + [''])

                # Add total number of instances to the y-axis labels
                y_labels = [f'{self.dataset_info[clean_tier][i]} [n={int(row_sums[i].item()):,.0f}]'.replace(',', "'") for i in range(matrix.size(0))]
                ax.set_yticklabels(y_labels + [''])

                ax.set_xlabel('Predictions')
                ax.set_ylabel('Targets')

                # Move x-axis label and ticks to the top
                ax.xaxis.set_label_position('top')
                ax.xaxis.set_ticks_position('top')

                fig.tight_layout()

                for i in range(matrix.size(0)):
                    for j in range(matrix.size(1)):
                        if zero_rows[i]:
                            ax.text(j, i, 'N/A', ha='center', va='center', color='black')
                        else:
                            ax.text(j, i, f'{matrix_percent[i, j]:.2f}', ha='center', va='center', color='#F88379', weight='bold') # coral red
                trainer.logger.experiment.log({f"{phase}_{tier}_confusion_matrix_{mode}": wandb.Image(fig)})
                plt.close()
                confusion_matrix.reset()

class LogMessisMetrics(pl.Callback):
    def __init__(self, hparams, dataset_info_file, debug=False):
        super().__init__()

        assert hparams.get('tiers') is not None, "Tiers must be defined in the hparams"

        self.tiers_dict = hparams.get('tiers')
        self.tiers = ['tier1', 'tier2', 'tier3']
        self.phases = ['train', 'val', 'test']
        self.modes = ['pixelwise', 'majority']
        self.debug = debug

        if debug:
            print(f"Phases: {self.phases}, Tiers: {self.tiers}, Modes: {self.modes}")

        with open(dataset_info_file, 'r') as f:
            self.dataset_info = json.load(f)

        # Initialize metrics
        self.metrics_to_compute = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa']
        self.metrics = {phase: {tier: {mode: self.__init_metrics(tier, phase) for mode in self.modes} for tier in self.tiers} for phase in self.phases}
        self.images_to_log = {phase: {mode: None for mode in self.modes} for phase in self.phases}
        self.images_to_log_targets = {phase: None for phase in self.phases}
        self.field_ids_to_log_targets = {phase: None for phase in self.phases}
        self.inputs_to_log = {phase: None for phase in self.phases}

    def __init_metrics(self, tier, phase):
        num_classes = self.tiers_dict[tier]['num_classes']

        accuracy = classification.MulticlassAccuracy(num_classes=num_classes, average='macro')
        per_class_accuracies = {
            class_index: classification.MulticlassAccuracy(num_classes=num_classes, average='macro') for class_index in range(num_classes)
        }
        precision = classification.MulticlassPrecision(num_classes=num_classes, average='macro')
        recall = classification.MulticlassRecall(num_classes=num_classes, average='macro')
        f1 = classification.MulticlassF1Score(num_classes=num_classes, average='macro')
        cohen_kappa = classification.MulticlassCohenKappa(num_classes=num_classes)

        return {
            'accuracy': accuracy,
            'per_class_accuracies': per_class_accuracies,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cohen_kappa': cohen_kappa
        }

    def setup(self, trainer, pl_module, stage=None):
        # Move all metrics to the correct device at the start of the training/validation
        device = pl_module.device
        for phase_metrics in self.metrics.values():
            for tier_metrics in phase_metrics.values():
                for mode_metrics in tier_metrics.values():
                    for metric in self.metrics_to_compute:
                        mode_metrics[metric].to(device)
                    for class_accuracy in mode_metrics['per_class_accuracies'].values():
                        class_accuracy.to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'test')

    def __on_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, phase):
        if trainer.sanity_checking:
            return
        if self.debug:
            print(f"{phase} batch ended. Updating metrics...")

        targets = torch.stack(batch[1][0]) # (tiers, batch, H, W)
        outputs = outputs['outputs'][-1] # (batch, H, W)        
        field_ids = batch[1][1].permute(1, 0, 2, 3)[0]

        pixelwise_outputs, majority_outputs = LogConfusionMatrix.get_pixelwise_and_majority_outputs(outputs, field_ids, self.dataset_info)        

        for preds, mode in zip([pixelwise_outputs, majority_outputs], self.modes):

            # Update all metrics
            assert preds.shape == targets.shape, f"Shapes of predictions and targets do not match: {preds.shape} vs {targets.shape}"
            assert preds.shape[0] == len(self.tiers), f"Number of tiers in predictions and tiers do not match: {preds.shape[0]} vs {len(self.tiers)}"
           
            self.images_to_log[phase][mode] = preds[-1]
            
            for pred, target, tier in zip(preds, targets, self.tiers):
                # flatten and remove background class if the mode is majority (such that the background class is not considered in the metrics)
                if mode == 'majority':
                    pred = pred[target != 0]
                    target = target[target != 0]
                metrics = self.metrics[phase][tier][mode]
                for metric in self.metrics_to_compute:
                    metrics[metric].update(pred, target)
                    if self.debug:
                        print(f"{phase} {tier} {mode} {metric} updated. Update count: {metrics[metric]._update_count}")
                self.__update_per_class_accuracy(pred, target, metrics['per_class_accuracies'])

        self.images_to_log_targets[phase] = targets[-1]
        self.field_ids_to_log_targets[phase] = field_ids
        self.inputs_to_log[phase] = batch[0]

    def __update_per_class_accuracy(self, preds, targets, per_class_accuracies):
        for class_index, class_accuracy in per_class_accuracies.items():
            class_mask = targets == class_index
            if class_mask.any():
                class_accuracy.update(preds[class_mask], targets[class_mask])
                if self.debug:
                    print(f"Per-class accuracy for class {class_index} updated. Update count: {class_accuracy._update_count}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__on_epoch_end(trainer, pl_module, 'train')

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__on_epoch_end(trainer, pl_module, 'val')

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__on_epoch_end(trainer, pl_module, 'test')

    def __on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, phase):
        if trainer.sanity_checking:
            return # Skip during sanity check (avoid warning about metric compute being called before update)
        for tier in self.tiers:
            for mode in self.modes:
                metrics = self.metrics[phase][tier][mode]

                # Calculate and reset in tier: Accuracy, Precision, Recall, F1, Cohen's Kappa
                metrics_dict = {metric: metrics[metric].compute() for metric in self.metrics_to_compute}
                pl_module.log_dict({f"{phase}_{metric}_{tier}_{mode}": v for metric, v in metrics_dict.items()}, on_step=False, on_epoch=True)
                for metric in self.metrics_to_compute:
                    metrics[metric].reset()

                class_names_mapping = self.dataset_info[tier.split('_')[0] if '_refined' in tier else tier] 

                class_accuracies = []
                for class_index, class_accuracy in metrics['per_class_accuracies'].items():
                    if class_accuracy._update_count == 0:
                        continue  # Skip if no updates have been made
                    class_accuracies.append([class_index, class_names_mapping[class_index], class_accuracy.compute()])
                    class_accuracy.reset()
                wandb_table = wandb.Table(data=class_accuracies, columns=["Class Index", "Class Name", "Accuracy"])
                # Log the table
                trainer.logger.experiment.log({f"{phase}_per_class_accuracies_{tier}_{mode}": wandb_table})

        # use the same n_classes for all images, such that they are comparable
        n_classes = max([
            torch.max(self.images_to_log_targets[phase]),
            torch.max(self.images_to_log[phase]["majority"]),
            torch.max(self.images_to_log[phase]["pixelwise"])
        ])
        images     = [LogMessisMetrics.process_images(self.images_to_log[phase][mode], n_classes) for mode in self.modes]
        images.append(LogMessisMetrics.create_positive_negative_image(self.images_to_log[phase]["majority"], self.images_to_log_targets[phase]))
        images.append(LogMessisMetrics.process_images(self.images_to_log_targets[phase], n_classes))
        images.append(LogMessisMetrics.process_images(self.field_ids_to_log_targets[phase].cpu()))

        examples = []
        for i in range(len(images[0])):
            example = np.concatenate([img[i] for img in images], axis=0)
            examples.append(wandb.Image(example, caption=f"From Top to Bottom: {self.modes[0]}, {self.modes[1]}, right/wrong classifications, target, fields"))

        trainer.logger.experiment.log({f"{phase}_examples": examples})

        # Log segmentation masks
        batch_input_data = self.inputs_to_log[phase].cpu() # shape [BS, 6, 3, 224, 224]
        ground_truth_masks = self.images_to_log_targets[phase].cpu().numpy()
        pixel_wise_masks = self.images_to_log[phase]["pixelwise"].cpu().numpy()
        field_majority_masks = self.images_to_log[phase]["majority"].cpu().numpy()
        correctness_masks = self.create_positive_negative_segmentation_mask(field_majority_masks, ground_truth_masks)
        class_labels = {idx: name for idx, name in enumerate(self.dataset_info["tier3"])}

        segmentation_masks = []
        for input_data, ground_truth_mask, pixel_wise_mask, field_majority_mask, correctness_mask in zip(batch_input_data, ground_truth_masks, pixel_wise_masks, field_majority_masks, correctness_masks):
            middle_timestep_index = input_data.shape[1] // 2  # Get the middle timestamp index
            gamma = 2.5  # Gamma for brightness adjustment
            rgb_image = input_data[:3, middle_timestep_index, :, :].permute(1, 2, 0).numpy()  # Shape [224, 224, 3]
            rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
            rgb_image = np.power(rgb_image, 1.0 / gamma)
            rgb_image = (rgb_image * 255).astype(np.uint8)

            mask_img = wandb.Image(
                rgb_image,
                masks={
                    "predictions_pixel_wise": {"mask_data": pixel_wise_mask, "class_labels": class_labels},
                    "predictions_field_majority": {"mask_data": field_majority_mask, "class_labels": class_labels},
                    "ground_truth": {"mask_data": ground_truth_mask, "class_labels": class_labels},
                    "correctness": {"mask_data": correctness_mask, "class_labels": { 0: "Background", 1: "Wrong", 2: "Right" }},
                },
            )
            segmentation_masks.append(mask_img)

        trainer.logger.experiment.log({f"{phase}_segmentation_mask": segmentation_masks})

        if self.debug:
            print(f"{phase} epoch ended. Logging & resetting metrics...", trainer.sanity_checking)

    @staticmethod
    def create_positive_negative_segmentation_mask(field_majority_masks, ground_truth_masks):
        """
        Create a tensor that shows the positive and negative classifications of the model.

        Args:
            field_majority_masks (np.ndarray): The field majority masks generated by the model.
            ground_truth_masks (np.ndarray): The ground truth masks.

        Returns:
            np.ndarray: An array with values:
                - 0 where the target is 0,
                - 2 where the prediction matches the target,
                - 1 where the prediction does not match the target.
        """
        correctness_mask = np.zeros_like(ground_truth_masks, dtype=int)

        matches = (field_majority_masks == ground_truth_masks) & (ground_truth_masks != 0)
        correctness_mask[matches] = 2

        mismatches = (field_majority_masks != ground_truth_masks) & (ground_truth_masks != 0)
        correctness_mask[mismatches] = 1

        return correctness_mask

    @staticmethod
    def create_positive_negative_image(generated_images, target_images):
        """
        Create an image that shows the positive and negative classifications of the model.

        Args:
            generated_images (torch.Tensor): The images generated by the model.
            target_images (torch.Tensor): The target images.

        Returns:
            list: A list of processed images.
        """
        classification_masks = generated_images == target_images
        processed_imgs = []
        for mask, target in zip(classification_masks, target_images):
            # color the background white, right classifications green, wrong classifications red
            colored_img = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.uint8)
            mask = mask.bool()  # Convert to boolean tensor
            colored_img[mask] = torch.tensor([0, 255, 0], dtype=torch.uint8)
            colored_img[~mask] = torch.tensor([255, 0, 0], dtype=torch.uint8)
            colored_img[target == 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)
            processed_imgs.append(colored_img.cpu())
        return processed_imgs

    @staticmethod
    def process_images(imgs, max=None):
        """
        Process a batch of images to be logged on wandb.

        Args:
            imgs (torch.Tensor): A batch of images with shape (B, H, W) to be processed.
            max (float, optional): The maximum value to normalize the images. Defaults to None. If None, the maximum value in the batch is used.
        """
        if max is None:
            max = np.max(imgs.cpu().numpy())
        normalized_img = imgs / max
        processed_imgs = []
        for img in normalized_img.cpu().numpy():
            if max < 60:
                cmap = ListedColormap(plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors)
            else:
                cmap = plt.get_cmap('viridis')
            colored_img = cmap(img)
            colored_img[img == 0] = [0, 0, 0, 1]
            colored_img_uint8 = (colored_img[:, :, :3] * 255).astype(np.uint8)
            processed_imgs.append(colored_img_uint8)
        return processed_imgs