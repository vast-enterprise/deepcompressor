import peft.tuners.lora.layer as lora
import torch.nn as nn

def is_linear(module)->bool:
    return isinstance(module, (lora.Linear, nn.Linear))