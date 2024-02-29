import os
import argparse
from datetime import datetime
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from train_module import TrainModule
from utils import load_config


def main(config_dir: str, **kargs: Dict[str, Any]) -> None:
    general_config_path = os.path.join(config_dir, "general_config.yaml")
    model_config_path = os.path.join(config_dir, "model_config.yaml")
    train_config_path = os.path.join(config_dir, "train_config.yaml")

    general_config_dict = load_config(general_config_path)
    model_config_dict = load_config(model_config_path)
    train_config_dict = load_config(train_config_path)

    pl.seed_everything(train_config_dict.get("setting").get("random_seed"))
    torch.set_float32_matmul_precision("medium")

    data_module = DataModule(config=train_config_dict)
    data_module.prepare_data()
    data_module.setup()

    train_module = TrainModule(
        train_config=train_config_dict,
        model_config=model_config_dict,
        input_size=len(data_module.features),
        seq_len=data_module.max_seq_len,
    )

    tensorboard_config = general_config_dict.get("tensorboard")
    logger = TensorBoardLogger(
        save_dir=tensorboard_config.get("save_dir", "./tensorboard"),
        name=tensorboard_config.get("exp_name"),
    )
    logger.log_hyperparams(general_config_dict)
    logger.log_hyperparams(train_config_dict)
    logger.log_hyperparams(model_config_dict)

    trainer = pl.Trainer(
        max_epochs=train_config_dict.get("trainer").get("max_epochs"),
        accelerator="auto",
        log_every_n_steps=train_config_dict.get("trainer").get("log_every_n_step"),
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="valid_loss",
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode="min",
            )
        ],
    )

    trainer.fit(model=train_module, datamodule=data_module)
    trainer.test(model=train_module, datamodule=data_module)

    save_path = os.path.join(
        logger.log_dir,
        f"{train_config_dict.get('setting').get('model_name')}.pth",
    )
    torch.jit.save(train_module.to_torchscript(), save_path)


if __name__ == "__main__":
    curr_time_str = datetime.now().strftime("%Y%m%d_%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_dir",
        type=str,
        dest="config_dir",
        default="./configs",
        help="config directory path",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        dest="exp_name",
        default=f"{curr_time_str}_exp",
        help="current experiement's name",
    )
    args_dict = vars(parser.parse_args())

    main(config_dir=args_dict.get("config_dir"))