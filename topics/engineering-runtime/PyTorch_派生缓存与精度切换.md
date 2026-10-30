---
title: PyTorch 派生缓存的精度转换与失效处理
created: 2026-10-24
updated: 2026-10-30
status: ready
blog: true
tags: [PyTorch, FP16, 故障定位]
---

# PyTorch 派生缓存的精度转换与失效处理

模型参数已经转换成 FP16，推理却仍然报 Half 与 Float 不匹配。检查参数列表时看不出异常，因为留下 FP32 数据的对象，是评估模式下生成的一份派生缓存。

这个问题最早表现为缓存没有跟随模型迁移设备，后续精度检查又遇到了相同类型的故障：先在 FP32 下调用 `eval()`，再调用 `.half()`，缓存保留原来的 dtype。对该实现，先转换精度、再调用 `eval()` 重建缓存，可以恢复正常执行。问题涉及模块状态的管理，与模型计算公式无关。

# 一、参数之外还有哪些张量

PyTorch 模块中的 Tensor 不一定都是参数。对精度和设备转换而言，三类对象的行为不同：

| 对象 | 是否随模块执行精度和设备转换 | 是否进入 state_dict |
| --- | --- | --- |
| Parameter | 是，按转换接口规则处理 | 是 |
| 注册的 buffer | 是，浮点 buffer 可转换精度 | 默认进入，可设为非持久 |
| 普通属性中的 Tensor | 不由模块转换接口自动遍历处理 | 不自动进入 |

因此，`model.parameters()` 中所有参数都正确，仍不足以证明下一次前向使用的张量都正确。模块可能从参数展开索引表、生成位置偏置、构造掩码或缓存其他中间结果，再直接挂到普通属性上。

这些对象参与后续计算，却没有进入参数或 buffer 的管理范围。检查它们时，需要沿前向实际读取的属性回溯，而不是只遍历权重。

# 二、调用顺序为什么改变结果

下面用一个与具体模型无关的例子说明缓存类型的变化，不包含实际网络实现：

```python
import torch
from torch import nn

class CachedScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if hasattr(self, "cached_scale"):
                del self.cached_scale
        else:
            with torch.no_grad():
                self.cached_scale = self.scale * 2
        return self

m = CachedScale().eval()  # 此时生成 FP32 缓存
m.half()                 # 参数转换，普通 Tensor 属性保留
assert m.scale.dtype == torch.float16
assert m.cached_scale.dtype == torch.float32

m.eval()                 # 该实现会重新生成缓存
assert m.cached_scale.dtype == torch.float16
```

`eval()` 等价于调用 `train(False)`。普通模块只需要切换训练标志，但自定义模块可以在重写的 `train()` 中做更多事情。这里的缓存刷新正是这种附加行为，所以再次调用 `eval()` 才有作用。

实际故障中，FP16 运算结果与 FP32 缓存相加后可能发生类型提升，直到后面的矩阵乘法才报错。报错算子暴露了冲突，缓存生成的位置才解释了冲突从哪里来。直接在最后一个矩阵乘法前强制转换，容易掩盖更早的状态问题。

对应验证流程调整为先完成设备和精度转换，再进入评估模式。这个顺序适用于会在 `eval()` 中重建缓存的实现；若模块没有刷新逻辑，重复调用 `eval()` 并不会自动转换普通属性。

# 三、缓存随模型迁移还不够

对于无需训练、可以重建的张量，非持久 buffer 是一种常见选择：

```python
self.register_buffer("cached_value", None, persistent=False)
```

在缓存被赋予 Tensor 后，它能够参与模块的设备与精度转换，同时不写入 `state_dict`。`persistent=False` 控制的是序列化范围，不是是否迁移。

但注册 buffer 不能自动解决数据过期。例如，缓存由某个参数计算而来，之后加载了新权重，旧缓存仍可能保留；设备和 dtype 都正确，数值却已经不再对应当前参数。

缓存需要同时考虑两种有效性：一是能否与当前输入执行计算，涉及 device、dtype 和 shape；二是内容是否仍对应当前参数和模式。前一种失效常表现为异常，后一种可能安静地产生错误结果。

失效处理应由模块明确负责：加载权重后清除派生缓存，进入需要缓存的模式时重建，或在前向中检查必要条件后惰性生成。若缓存还依赖输入尺寸，尺寸也应纳入检查。具体选择取决于重建成本和调用方式，不能只靠应用层记住某个调用顺序。

# 四、验证应覆盖状态变化

一次冷启动前向通过，覆盖不了已生成缓存之后的迁移。更有效的检查是先触发缓存，再转换精度或设备，确认前向读取的对象与当前模型一致。

同样需要检查训练与评估往返，以及已有缓存时加载新权重。加载后的输出应与同权重的全新模型比较；只检查缓存存在、shape 正确、参数加载成功，无法发现数值过期。

这些检查可以放在隔离副本中完成。测试的是模块的状态转换契约，不需要修改训练策略。若准备继续导出模型，还应先固定评估状态与缓存内容，再按[ONNX 导出中的 Python 形状计算与静态适配](../graphs-operators/ONNX_形状计算与静态适配.md)检查图捕获过程。

# 五、参考资料

- [PyTorch Module：half、eval、register_buffer 与 load_state_dict](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html)
