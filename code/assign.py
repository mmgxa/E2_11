import argparse

import os
import numpy as np
import json
import logging
from datetime import datetime

from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.models import resnet18, ResNet18_Weights

from pytorch_lightning import Trainer, LightningDataModule, LightningModule, Callback

from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


# from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 10

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

        
class CIFAR10DataModule(LightningDataModule):

    def __init__(
        self,
        val_test_split: Tuple[int, int] = (5_000, 5_000),
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        
        # data transformations
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        CIFAR10(root='.', train=True, download=True)
        CIFAR10(root='.', train=False, download=True)


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CIFAR10(root='.', train=True, transform=self.train_transform)
            testset = CIFAR10(root='.', train=False, transform=self.test_transform)
            self.data_val, self.data_test = random_split(
                dataset=testset,
                lengths=self.hparams.val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
class CIFAR10LitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        
        self.example_input_array = torch.rand((1,3,32,32))
        self.tb_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        )
                
    def forward(self, x: torch.Tensor):
        return self.net(x)
        
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # add all parameters to a histogram
        self.logger.experiment.add_scalar("Accuracy/Train", self.train_acc.compute(), self.current_epoch)
        
        data_train = CIFAR10(root='.', train=True, transform=self.tb_transform)
        dl = DataLoader(
                dataset=data_train,
                batch_size=25,
            )
        images = (next(iter(dl)))[0]
        self.logger.experiment.add_images(f"25_training_data_examples_for_epoch_{self.current_epoch}",
                                        images,
                                        global_step=self.current_epoch,
                                        )
        
        
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()
        

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.logger.experiment.add_scalar("Accuracy/Val", acc, self.current_epoch)
        self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.logger.experiment.add_scalar("Accuracy/Test", self.test_acc.compute(), self.current_epoch)
        self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return {
            "optimizer": torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
        }

def train(args):

    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc=torch.nn.Linear(512,10)
    logger = TensorBoardLogger(save_dir=args.logdir, name=f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}", log_graph=True)

    
    datamodule = CIFAR10DataModule(batch_size=args.batch_size,
                                num_workers=8)
    model = CIFAR10LitModule(net=net)
            
    # to print/log environment variables

        
    trainer = Trainer(
        max_epochs=args.max_epochs,
        devices=1, 
        logger=logger,
        num_sanity_val_steps=0,
        accelerator=DEVICE,
        log_every_n_steps=1
        )

    LOGGER.info("Training model...")
    trainer.fit(model, datamodule)

    LOGGER.info("Saving model...")
    torch.save(net.state_dict(), "model.pth")
    


def parse_args():
    # SageMaker passes hyperparameters  as command-line arguments to the script
    # Parsing them below...
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logdir", type=str, default="logs/train_data/")
    args, _ = parser.parse_known_args()
    
    return args


if __name__ == "__main__":

    args = parse_args()   

    train(args)
