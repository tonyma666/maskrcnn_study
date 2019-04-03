# Time: 2019-04-03   23:19

from collections import namedtuple # 可以通过key进行索引的tuple

import torch
import torch.nn.functional as F
from torch import nn 

from maskrcnn_benchmark.layers import FrozenBatchNorm2d