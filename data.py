import os, sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from utils import load_data, stratified_split, pad_series


class CustomDataset(Dataset):
    def __init__(
        self,
        df_list: List[pd.DataFrame],
        features: List[str],
        class_col: str,
        num_class: int,
        seq_len: int,
        scaler_name: str,
    ) -> None:
        self.df_list = df_list
        self.features = features
        self.class_col = class_col
        self.num_class = num_class
        self.seq_len = seq_len
        self.scaler_name = scaler_name

    def __len__(self) -> int:
        return len(self.df_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return {
            "X": torch.tensor(
                pad_series(
                    self.df_list[index][self.features].values.astype(np.float32),
                    self.seq_len,
                ),
            ),
            "y": torch.tensor(int(self.df_list[index][self.class_col].iloc[0])).float(),
        }


class DataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        data_config = config.get("data")
        setting_config = config.get("setting")
        trainer_config = config.get("trainer")

        self.data_path = data_config.get("path")
        self.features = data_config.get("features")
        self.key_cols = data_config.get("key_cols")
        self.class_col = data_config.get("class_col")
        self.num_class = data_config.get("num_class")

        self.scaler_name = setting_config.get("scaler_name")
        self.test_size = setting_config.get("test_size", 0.2)
        self.batch_size = setting_config.get("batch_size", 32)
        self.random_seed = setting_config.get("random_seed", 42)

        self.shuffle = trainer_config.get("shuffle", False)
        self.num_workers = trainer_config.get("num_workers")

        self.dataset_class = getattr(
            sys.modules[__name__], setting_config.get("dataset_name")
        )

    def prepare_data(self) -> None:
        extension = os.path.splitext(self.data_path)[1].lstrip(".")
        total_df = load_data(self.data_path, extension)
        gb = total_df.groupby(self.key_cols)
        df_list = [df for (_, df) in gb]
        label_list = [int(df[self.class_col].iloc[0]) for (_, df) in gb]
        self.max_seq_len = max([len(df) for (_, df) in gb])

        (
            train_valid_df_list,
            self.test_df_list,
            train_valid_label_list,
            self.test_label_list,
        ) = stratified_split(df_list, label_list, self.test_size, self.random_seed)

        (
            self.train_df_list,
            self.valid_df_list,
            self.train_label_list,
            self.valid_label_list,
        ) = stratified_split(
            train_valid_df_list,
            train_valid_label_list,
            self.test_size,
            self.random_seed,
        )

    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(
            self.train_df_list,
            self.features,
            self.class_col,
            self.num_class,
            self.max_seq_len,
            self.scaler_name,
        )
        self.valid_dataset = self.dataset_class(
            self.valid_df_list,
            self.features,
            self.class_col,
            self.num_class,
            self.max_seq_len,
            self.scaler_name,
        )
        self.test_dataset = self.dataset_class(
            self.test_df_list,
            self.features,
            self.class_col,
            self.num_class,
            self.max_seq_len,
            self.scaler_name,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )