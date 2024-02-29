import os
import argparse
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data import DataModule
from utils import load_config


def main(
    model_name: str, exp_name: str, version_num: int, **kargs: Dict[str, Any]
) -> None:
    log_dir = f"./output/{exp_name}/version_{version_num}"
    model_path = os.path.join(log_dir, f"{model_name}.pth")
    config_path = os.path.join(log_dir, "hparams.yaml")

    config = load_config(config_path)

    train_config_dict = {
        "data": config.get("data"),
        "setting": config.get("setting"),
        "trainer": config.get("trainer"),
    }

    pl.seed_everything(train_config_dict.get("setting").get("random_seed"))

    data_module = DataModule(config=train_config_dict)
    data_module.prepare_data()
    data_module.setup()

    loaded_model = torch.jit.load(model_path)

    # customizing is needed below
    for input, real in DataLoader(data_module.test_dataset, batch_size=1):
        pred, attn_weight = loaded_model(input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        dest="model_name",
        help="model name to load",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        dest="exp_name",
        help="experiement name to load",
    )
    parser.add_argument(
        "-v",
        "--version_num",
        type=int,
        dest="version_num",
        help="version number to load",
    )
    args_dict = vars(parser.parse_args())

    main(
        model_name=args_dict.get("model_name"),
        exp_name=args_dict.get("exp_name"),
        version_num=args_dict.get("version_num"),
    )