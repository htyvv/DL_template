import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from abc import abstractmethod

from typing import Tuple, Dict, Any, List


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.core_layer = None
        self.output_layer = None

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self):
        """Model prints with number of trainable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return f"{super().__str__()}\nTrainable parameters: {params}"


class RNNBase(BaseModel):
    def __init__(self, config: Dict[str, Any], input_size: int, seq_len: int) -> None:
        super().__init__(config)

        self.input_size: int = input_size
        self.seq_len: int = seq_len
        self.output_size: int = config.get("output_size")
        self.hidden_size: int = config.get("hidden_size")
        self.num_layers: int = config.get("num_layers")
        self.bias: bool = config.get("bias", True)
        self.dropout: float = config.get("dropout", 0.0)
        self.bidirectional: bool = config.get("bidirectional", False)
        self.self_attention: bool = config.get("self_attention", False)

        self.n_directions: int = 2 if self.bidirectional else 1

        self.post_layer = nn.Linear(
            self.hidden_size * self.n_directions, self.input_size
        )
        self.self_attn_layer = SelfAttention(self.input_size)
        self.output_layer = nn.Linear(self.input_size * seq_len, self.output_size)

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        out, _ = self.core_layer(batch)

        out = self.post_layer(out)

        attn_weight = torch.empty(0)
        if self.self_attention:
            out, attn_weight = self.self_attn_layer(out)

        out = torch.flatten(out, 1)
        out = self.output_layer(out)

        return out, attn_weight


class CNNBase(BaseModel):
    def __init__(self, config: Dict[str, Any], input_size: int, seq_len: int):
        super().__init__(config)

        self.input_size: int = input_size
        self.seq_len: int = seq_len
        self.output_size: int = config.get("output_size")
        self.num_channels: List[int] = config.get("num_channels")
        self.kernel_size: int = config.get("kernel_size")
        self.self_attention: bool = config.get("self_attention", False)

    @abstractmethod
    def forward(self, batch: Tensor):
        pass


class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()

        self.q_layer = nn.Linear(input_size, input_size)
        self.k_layer = nn.Linear(input_size, input_size)
        self.v_layer = nn.Linear(input_size, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        query = self.q_layer(batch).transpose(1, 2).contiguous()
        key = self.k_layer(batch)
        value = self.v_layer(batch)
        energy = torch.bmm(query, key)
        attention_weight = self.softmax(energy)
        out = torch.bmm(value, attention_weight)

        return out, attention_weight