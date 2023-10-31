import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.utilities.model_helpers import get_torchvision_model
from torchmetrics import Accuracy
from typing import Optional

# Model:
class ImageNetLightningModel(LightningModule):

    def __init__(
        self,
        arch: str = "resnet18",
        weights: Optional[str] = None,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        workers: int = 4,
    ):
        super().__init__()
        self.arch = arch
        self.weights = weights
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.model = get_torchvision_model(self.arch, weights=self.weights)
        self.train_acc1 = Accuracy(task="multiclass", num_classes=1000, top_k=1)
        self.eval_acc1 = Accuracy(task="multiclass", num_classes=1000, top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        # update metrics
        self.train_acc1(output, target)
        self.log("train_acc1", self.train_acc1, prog_bar=True)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        # update metrics
        self.eval_acc1(output, target)
        self.log(f"{prefix}_acc1", self.eval_acc1, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transforms = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_data = tv.datasets.ImageFolder(root='data/train', transform=transforms)

        return torch.utils.data.DataLoader(
            train_data, 
            batch_size=256,
            num_workers=16,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
            )

    def val_dataloader(self):
        transforms = tv.transforms.Compose([
            tv.transforms.Resize(256), 
            tv.transforms.CenterCrop(224), 
            tv.transforms.ToTensor(),
        ])

        val_data = tv.datasets.ImageFolder(root='data/val', transform=transforms)

        return torch.utils.data.DataLoader(
            val_data, 
            batch_size=500,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True
            )

    def test_dataloader(self):
        return self.val_dataloader()


# Train:
if __name__ == "__main__":
    model = ImageNetLightningModel()
    trainer = Trainer(
        max_epochs=90,
        accelerator="auto",
        precision=32, # "16-mixed", "bf16-mixed", "transformer-engine", "64-true”,  "bf16-true”
        devices="auto",
        strategy="auto",  # "fsdp" "deepspeed_stage_x" "deepspeed_stage_2_offload" "ddp"
        profiler=None, #"advanced", "simple"
        logger=False,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            ModelCheckpoint(monitor="val_acc1", mode="max"),
        ],
    )
    trainer.fit(model)