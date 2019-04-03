# 把import写在__init__里面的好处就是在这里进行引用的包，外部调用就可以垮阶段了
# 比如: from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from .batch_norm import FrozenBatchNorm2d