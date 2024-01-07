import os
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, AUROC


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                os.path.join(self.data_dir, "train"), self.transform
            )
            self.val_dataset = ImageFolder(
                os.path.join(self.data_dir, "val"), self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=24)


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.learning_rate = learning_rate

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.train_f1 = F1Score(num_classes=num_classes)
        self.val_f1 = F1Score(num_classes=num_classes)
        self.train_auroc = AUROC(num_classes=num_classes)
        self.val_auroc = AUROC(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)

        # Calculate and log additional metrics
        preds = torch.argmax(logits, dim=1)
        self.log("train_acc", self.train_accuracy(preds, y))
        self.log("train_f1", self.train_f1(preds, y))
        self.log("train_auroc", self.train_auroc(F.softmax(logits, dim=1), y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

        # Calculate and log additional metrics
        preds = torch.argmax(logits, dim=1)
        self.log("val_acc", self.val_accuracy(preds, y))
        self.log("val_f1", self.val_f1(preds, y))
        self.log("val_auroc", self.val_auroc(F.softmax(logits, dim=1), y))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    # Set data directory
    data_dir = "/media/hdd3/neo/topviews_1k_split"

    # Set the number of classes in your dataset
    num_classes = 4

    # Initialize the data module and model
    data_module = ImageDataModule(data_dir)
    model = ResNetClassifier(num_classes=num_classes)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=100, devices=3, accelerator="gpu")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
