import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import classification
import wandb

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
    def __init__(self, num_classes_tier1, num_classes_tier2, num_classes_tier3, img_size=256, patch_size=16, num_frames=3, bands=[0, 1, 2, 3, 4, 5], weight_tier1=1.0, weight_tier2=1.0, weight_tier3=1.0, weight_tier3_refined=1.0, debug=False):
        super(HierarchicalClassifier, self).__init__()

        self.embed_dim=768
        self.num_frames=num_frames
        self.output_embed_dim = self.embed_dim * self.num_frames
        self.hp, self.wp = img_size // patch_size, img_size // patch_size
        self.head_channels = 256 # TODO: We should research what makes sense here (same channels, gradual decrease from 1024, ...)
        self.total_classes = num_classes_tier1 + num_classes_tier2 + num_classes_tier3
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
            pretrained='./prithvi/models/Prithvi_100M.pt',
            debug=self.debug
        )

        # Freeze the base model
        for param in self.prithvi.parameters():
            param.requires_grad = False

        # Neck to transform the token-based output of the transformer into a spatial feature map
        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=self.embed_dim * self.num_frames,
            output_embed_dim=self.output_embed_dim,
            drop_cls_token=True,
            Hp=self.hp,
            Wp=self.wp,
        )

        # Loss weights for the different tiers
        self.weight_tier1 = weight_tier1
        self.weight_tier2 = weight_tier2
        self.weight_tier3 = weight_tier3
        self.weight_tier3_refined = weight_tier3_refined

        # Hierarchical heads to predict tier 1, tier 2 and tier 3 classes
        self.head_tier1 = HierarchicalFCNHead(
            in_channels=self.output_embed_dim,
            out_channels=self.head_channels,
            num_classes=num_classes_tier1,
            num_convs=1,
            dropout_p=0.1,
            debug=self.debug
        )
        self.head_tier2 = HierarchicalFCNHead(
            in_channels=self.head_channels, # Match output from head_tier1
            out_channels=self.head_channels,
            num_classes=num_classes_tier2,
            num_convs=1,
            dropout_p=0.1,
            debug=self.debug
        )
        self.head_tier3 = HierarchicalFCNHead(
            in_channels=self.head_channels, # Match output from head_tier2
            out_channels=self.head_channels,
            num_classes=num_classes_tier3,
            num_convs=1,
            dropout_p=0.1,
            debug=self.debug
        )
        self.refinement_head_tier3 = LabelRefinementHead(input_channels=self.total_classes, num_classes=num_classes_tier3)

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

        # Remove from tuple
        features = features[0]
        if self.debug:
            print(f"Features shape after removing tuple: {safe_shape(features)}")

        # Process through the first tier
        output_tier1, features = self.head_tier1(features)

        if self.debug:    
            print(f"Features shape after tier 1 head: {safe_shape(features)}")
            print(f"Output shape after tier 1 head: {safe_shape(output_tier1)}")

        # Process through the second tier
        output_tier2, features = self.head_tier2(features)

        if self.debug:
            print(f"Features shape after tier 2 head: {safe_shape(features)}")
            print(f"Output shape after tier 2 head: {safe_shape(output_tier2)}")

        # Process through the third tier
        output_tier3, features = self.head_tier3(features)

        if self.debug:
            print(f"Features shape after tier 3 head: {safe_shape(features)}")
            print(f"Output shape after tier 3 head: {safe_shape(output_tier3)}")

        # Process through the classification refinement head
        output_concatenated = torch.cat([output_tier1, output_tier2, output_tier3], dim=1)
        output_tier3_refined = self.refinement_head_tier3(output_concatenated)

        return (output_tier1, output_tier2, output_tier3, output_tier3_refined)
    
    def calculate_loss(self, outputs, targets):
        output_tier1, output_tier2, output_tier3, output_tier3_refined = outputs
        target_tier1, target_tier2, target_tier3 = targets

        loss_tier1 = self.loss_func(output_tier1, target_tier1)
        loss_tier2 = self.loss_func(output_tier2, target_tier2)
        loss_tier3 = self.loss_func(output_tier3, target_tier3)
        loss_tier3_refined = self.loss_func(output_tier3_refined, target_tier3)

        return loss_tier1 * self.weight_tier1 + loss_tier2 * self.weight_tier2 + loss_tier3 * self.weight_tier3 + loss_tier3_refined * self.weight_tier3_refined
 
class Messis(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = HierarchicalClassifier(
            num_classes_tier1=hparams['tiers']['tier1']['num_classes'],
            num_classes_tier2=hparams['tiers']['tier2']['num_classes'],
            num_classes_tier3=hparams['tiers']['tier3']['num_classes'],
            img_size=hparams.get('img_size'),
            patch_size=hparams.get('patch_size'),
            num_frames=hparams.get('num_frames'),
            bands=hparams.get('bands'),
            weight_tier1=hparams['tiers']['tier1']['loss_weight'],
            weight_tier2=hparams['tiers']['tier2']['loss_weight'],
            weight_tier3=hparams['tiers']['tier3']['loss_weight'],
            weight_tier3_refined=hparams['tiers']['tier3_refined']['loss_weight'],
            debug=hparams.get('debug')
        )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, "val")
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.get('lr', 1e-3))
        return optimizer

    def __step(self, batch, batch_idx, stage):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.model.calculate_loss(outputs, targets)

        if self.hparams.get('debug'):
            print(f"Step Inputs shape: {safe_shape(inputs)}")
            print(f"Step Targets shape: {safe_shape(targets)}")
            print(f"Step Outputs shape: {safe_shape(outputs)}")

        # NOTE: All metrics other than loss are tracked by callbacks (LogMessisMetrics)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'outputs': outputs}
        
class LogConfusionMatrix(pl.Callback):
    def __init__(self, hparams):
        super().__init__()

        # Initialize confusion matrix for each tier
        self.confmat_tier1 = classification.MulticlassConfusionMatrix(num_classes=hparams['tiers']['tier1']['num_classes'])
        self.confmat_tier2 = classification.MulticlassConfusionMatrix(num_classes=hparams['tiers']['tier2']['num_classes'])
        self.confmat_tier3 = classification.MulticlassConfusionMatrix(num_classes=hparams['tiers']['tier3']['num_classes'])
        self.confmat_tier3_refined = classification.MulticlassConfusionMatrix(num_classes=hparams['tiers']['tier3_refined']['num_classes'])

    def setup(self, trainer, pl_module, stage=None):
        # Move all metrics to the correct device at the start of the training/validation
        device = pl_module.device
        self.confmat_tier1.to(device)
        self.confmat_tier2.to(device)
        self.confmat_tier3.to(device)
        self.confmat_tier3_refined.to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        actual_outputs = outputs['outputs']
        targets = batch[1]
        self.__update_confusion_matrices(actual_outputs, targets, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        actual_outputs = outputs['outputs']
        targets = batch[1]
        self.__update_confusion_matrices(actual_outputs, targets, pl_module)

    def __update_confusion_matrices(self, outputs, targets, pl_module):
        preds = [torch.softmax(out, dim=1).argmax(dim=1) for out in outputs]
        tiers = ['tier1', 'tier2', 'tier3', 'tier3_refined']
        target_tier1, target_tier2, target_tier3 = targets
        targets = [target_tier1, target_tier2, target_tier3, target_tier3]
        for pred, target, tier in zip(preds, targets, tiers):
            getattr(pl_module, f'confmat_{tier}')(pred, target)

    def on_train_epoch_end(self, trainer, pl_module):
        # Log and then reset the confusion matrices after training epoch
        self.__log_and_reset_confusion_matrices(trainer, pl_module, 'train')

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log and then reset the confusion matrices after validation epoch
        self.__log_and_reset_confusion_matrices(trainer, pl_module, 'val')

    def __log_and_reset_confusion_matrices(self, trainer, pl_module, phase):
        if trainer.sanity_checking:
            # Skip computing metrics during sanity check (avoid warning about metric compute being called before update)
            return
        matrices = {
            'tier1': pl_module.confmat_tier1.compute(),
            'tier2': pl_module.confmat_tier2.compute(),
            'tier3': pl_module.confmat_tier3.compute(),
            'tier3_refined': pl_module.confmat_tier3_refined.compute()
        }
        for tier, matrix in matrices.items():
            trainer.logger.experiment.log({
                f"{phase}_{tier}_confusion_matrix": self.create_wandb_confusion_matrix(matrix)
            })
            getattr(pl_module, f'confmat_{tier}').reset()

    @staticmethod
    def create_wandb_confusion_matrix(matrix):
        class_names = [f"Class {i}" for i in range(matrix.size(0))]
        preds, y_true = LogConfusionMatrix.prepare_matrix_data(matrix)
        return wandb.plot.confusion_matrix(probs=None, preds=preds, y_true=y_true, class_names=class_names)

    @staticmethod
    def prepare_matrix_data(matrix):
        preds, y_true = [], []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                count = int(matrix[i][j])
                preds.extend([j] * count)
                y_true.extend([i] * count)
        return preds, y_true
    
class LogMessisMetrics(pl.Callback):
    def __init__(self, hparams, debug=False):
        super().__init__()

        assert hparams.get('tiers') is not None, "Tiers must be defined in the hparams"

        self.tiers_dict = hparams.get('tiers')
        self.tiers = list(self.tiers_dict.keys())
        self.phases = ['train', 'val', 'test']
        self.debug = debug

        if debug: print(f"Phases: {self.phases}, Tiers: {self.tiers}")

        # Initialize metrics
        self.metrics_to_compute = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa']
        self.metrics = {phase: {tier: self.__init_metrics(tier, phase) for tier in self.tiers} for phase in self.phases}

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
                for metric in self.metrics_to_compute:
                    tier_metrics[metric].to(device)
                for class_accuracy in tier_metrics['per_class_accuracies'].values():
                    class_accuracy.to(device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.__on_batch_end(trainer, pl_module, outputs, batch, batch_idx, 'test')

    def __on_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, phase):
        if trainer.sanity_checking: return
        if self.debug: print(f"{phase} batch ended. Updating metrics...")

        outputs = outputs['outputs']
        tier1_targets, tier2_targets, tier3_targets = batch[1]
        targets = [tier1_targets, tier2_targets, tier3_targets, tier3_targets]
        preds = [torch.softmax(out, dim=1).argmax(dim=1) for out in outputs]

        # Update all metrics
        assert len(preds) == len(targets), f"Number of predictions and targets do not match: {len(preds)} vs {len(targets)}"
        assert len(preds) == len(self.tiers), f"Number of predictions and tiers do not match: {len(preds)} vs {len(self.tiers)}"
        for pred, target, tier in zip(preds, targets, self.tiers):
            
            metrics = self.metrics[phase][tier]
            for metric in self.metrics_to_compute:
                metrics[metric].update(pred, target)
                if self.debug: print(f"{phase} {tier} {metric} updated. Update count: {metrics[metric]._update_count}")
            self.__update_per_class_accuracy(pred, target, metrics['per_class_accuracies'])

    def __update_per_class_accuracy(self, preds, targets, per_class_accuracies):
        for class_index, class_accuracy in per_class_accuracies.items():
            class_mask = targets == class_index
            if class_mask.any():
                class_accuracy.update(preds[class_mask], targets[class_mask])
                if self.debug: print(f"Per-class accuracy for class {class_index} updated. Update count: {class_accuracy._update_count}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__on_epoch_end(trainer, pl_module, 'train')

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__on_epoch_end(trainer, pl_module, 'val')

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__on_epoch_end(trainer, pl_module, 'test')

    def __on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, phase):
        if trainer.sanity_checking: return # Skip during sanity check (avoid warning about metric compute being called before update)
        accuracies = []
        for tier in self.tiers:
            metrics = self.metrics[phase][tier]

            # Calculate and reset in tier: Accuracy, Precision, Recall, F1, Cohen's Kappa
            metrics_dict = {metric: metrics[metric].compute() for metric in self.metrics_to_compute}
            pl_module.log_dict({f"{phase}_{metric}_{tier}": v for metric, v in metrics_dict.items()}, on_step=False, on_epoch=True)
            for metric in self.metrics_to_compute:
                metrics[metric].reset()

            # Collect accuracies for overall accuracy calculation
            accuracies.append(metrics_dict['accuracy'])

            # Per-class accuracy in tier
            for class_index, class_accuracy in metrics['per_class_accuracies'].items():
                if class_accuracy._update_count == 0:
                    continue # Skip if no updates have been made (no samples of this class in the processed dataset partition)
                class_accuracy_value = class_accuracy.compute()
                pl_module.log(f"{phase}_accuracy_{tier}_class_{class_index}", class_accuracy_value, on_step=False, on_epoch=True)
                class_accuracy.reset()

        # Overall accuracy
        overall_accuracy = sum(accuracies) / len(accuracies)
        pl_module.log(f"{phase}_accuracy_overall", overall_accuracy, on_step=False, on_epoch=True)

        if self.debug: print(f"{phase} epoch ended. Logging & resetting metrics...", trainer.sanity_checking)