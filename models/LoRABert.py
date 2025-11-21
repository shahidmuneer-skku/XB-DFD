import torch
import torch.nn as nn

class LoRAForBERT(nn.Module):
    def __init__(self, linear: nn.Linear, linear_a: nn.Linear, linear_b: nn.Linear):
        super().__init__()
        self.linear = linear
        self.linear_a = linear_a
        self.linear_b = linear_b

    def forward(self, x):
        # Normal linear projection
        out = self.linear(x)
        # Low-rank residual
        lora_update = self.linear_b(self.linear_a(x))
        # Combine
        return out + lora_update
