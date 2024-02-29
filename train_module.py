from typing import Dict, Any

import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Recall, Precision, F1Score
from torchmetrics.regression import mae, mape, log_mse

import model


class TrainModule(pl.LightningModule):
    def __init__(
        self,
        train_config: Dict[str, Any],
        model_config: Dict[str, Any],
        input_size: int,
        seq_len: int,
    ):
        super().__init__()

        self.train_setting = train_config.get("setting")

        self.model_name = self.train_setting.get("model_name")
        self.task_type = self.train_setting.get("task_type")
        self.model = getattr(model, self.model_name)(
            model_config.get(self.model_name), input_size, seq_len
        )
        self.optimizer_class = getattr(optim, self.train_setting.get("optimizer"))
        self.loss_func = getattr(nn, self.train_setting.get("loss"))()

        self.scheduler_name = self.train_setting.get("scheduler").get("name")
        self.learning_rate = self.train_setting.get("learning_rate")

        if self.task_type != "regression":
            num_class = train_config.get("data").get("num_class")
            self.acc = Accuracy(
                task=self.task_type, num_classes=num_class, average="macro",
            )
            self.recall = Recall(
                task=self.task_type, num_classes=num_class, average="macro",
            )
            self.precision = Precision(
                task=self.task_type, num_classes=num_class, average="macro",
            )
            self.f1 = F1Score(
                task=self.task_type, num_classes=num_class, average="macro",
            )

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        configure_dict = {"optimizer": optimizer}

        if self.scheduler_name is not None:
            scheduler_params = self.train_setting.get("scheduler").get("params")
            scheduler_class = getattr(
                lr, self.train_setting.get("scheduler").get("name")
            )
            scheduler = scheduler_class(optimizer, **scheduler_params)
            configure_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
            }

        return configure_dict

    def training_step(self, batch, batch_idx):
        inputs = batch.get("X")
        target = batch.get("y")
        output, attn_weight = self.model(inputs)
        loss = self.loss_func(output, target)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )

        if self.task_type != "regression":
            self.log_dict(
                {
                    "train_acc": self.acc(output.argmax(1), target),
                    "train_recall": self.recall(output.argmax(1), target),
                    "train_precision": self.precision(output.argmax(1), target),
                    "train_f1": self.f1(output.argmax(1), target),
                },
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        if self.scheduler_name is not None:
            lr = torch.tensor(self.lr_schedulers().get_last_lr())
            self.log("learning_rate", lr, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch.get("X")
        target = batch.get("y")
        output, attn_weight = self.model(inputs)
        loss = self.loss_func(output, target)

        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )

        if self.task_type != "regression":
            self.log_dict(
                {
                    "valid_acc": self.acc(output.argmax(1), target),
                    "valid_recall": self.recall(output.argmax(1), target),
                    "valid_precision": self.precision(output.argmax(1), target),
                    "valid_f1": self.f1(output.argmax(1), target),
                },
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch.get("X")
        target = batch.get("y")
        output, attn_weight = self.model(inputs)
        loss = self.loss_func(output, target)
        self.log("test_loss", loss, on_epoch=True, logger=True, prog_bar=True)

        return loss