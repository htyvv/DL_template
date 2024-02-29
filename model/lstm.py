from torch import nn
from typing import Dict
from .base import RNNBase


class LSTM(RNNBase):
    def __init__(self, config: Dict[str, str], input_size: int, seq_len: int) -> None:
        super().__init__(config, input_size, seq_len)

        self.core_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )