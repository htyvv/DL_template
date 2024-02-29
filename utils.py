import yaml
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import torch

import sklearn.preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


def load_config(yml_path: str, encoding="UTF-8") -> Dict[str, Any]:
    with open(yml_path, "r", encoding=encoding) as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def load_data(file_path: str, extension: str) -> pd.DataFrame:
    try:
        return getattr(pd, f"read_{extension}")(file_path)
    except Exception as e:
        raise Exception(
            f"Pandas v.{pd.__version__} doesn't support extension: {extension}"
        )


def scale_data(
    scaler_name: str,
    target_df: pd.DataFrame,
    fitting_df: pd.DataFrame,
    scale_features: List[str],
    verbose: bool = True,
) -> None:
    """
    Scale selected features in a target DataFrame using a specified scaler from scikit-learn.

    Parameters:
    - scaler_name (str): The name of the scaler to be used (e.g., 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler').
    - target_df (pd.DataFrame): The DataFrame to be scaled.
    - fitting_df (pd.DataFrame): The DataFrame used for fitting the scaler.
    - scale_features (List[str]): List of feature names to be scaled.
    - verbose (bool, optional): If True, print a message indicating the applied scaler and features.

    Returns:
    - None

    This function scales the specified features in the target_df using the provided scaler, with criteria taken from criteria_df.
    The target_df is modified in place.

    Example usage:
    - scale_data('StandardScaler', my_data, criteria_data, ['feature1', 'feature2'], verbose=True)
    """
    scaler = getattr(sklearn.preprocessing, scaler_name)()
    scaler.fit(fitting_df.loc[:, scale_features])
    target_df.loc[:, scale_features] = scaler.transform(
        target_df.loc[:, scale_features]
    )

    if verbose:
        print(f"Applied {scaler_name} to {scale_features}")


def pad_series(input_arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad or truncate an input array to match the target length.

    Parameters:
    - input_arr (numpy.ndarray): The input array to be padded or truncated.
    - target_len (int): The desired target length of the output array.

    Returns:
    - numpy.ndarray: The padded or truncated array.
    """
    curr_len = len(input_arr)

    if curr_len < target_len:
        # Pad the 2d series with edge values to match the target length
        pad_width = target_len - curr_len
        output_arr = np.pad(input_arr, ((0, pad_width), (0, 0)), mode="edge")
    else:
        # Truncate or return the input 2d series as is
        output_arr = input_arr[:target_len, :]

    return output_arr


def f1_loss(y_true, y_pred, weight):
    tp = torch.sum((y_true * y_pred).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

    return 1 - (f1 @ weight) / weight.sum()


def stratified_split(df_list, label_list, test_size, random_state):
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idxes, test_idxes = next(sss.split(X=df_list, y=label_list))
    train_df_list = [df_list[i] for i in train_idxes]
    test_df_list = [df_list[i] for i in test_idxes]
    train_label_list = [label_list[i] for i in train_idxes]
    test_label_list = [label_list[i] for i in test_idxes]

    return train_df_list, test_df_list, train_label_list, test_label_list