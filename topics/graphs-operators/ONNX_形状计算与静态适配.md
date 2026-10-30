---
title: ONNX 导出中的 Python 形状计算与静态适配
created: 2026-10-30
updated: 2026-10-30
status: ready
blog: true
tags: [PyTorch, ONNX, ONNX Runtime, 模型导出, 精度对齐]
---

# ONNX 导出中的 Python 形状计算与静态适配

一个模块在 PyTorch 中可以正常执行，导出时却被根据输入尺寸调用的 Python `round()` 阻塞。出问题的表达并不复杂：从特征的长宽比计算池化输出尺寸，再将整数尺寸交给后续算子。

固定输入尺寸后，把这部分计算提前到图外，静态表达完成了导出与 ONNX Runtime 核对。这个案例涉及 Python 尺寸计算如何进入计算图，以及静态部署究竟需要保留哪些输入条件。

# 一、能在 Python 中运行不等于能捕获成图

普通 eager 执行时，尺寸往往表现为可供 Python 使用的整数。程序可以做比较、除法、取整，再决定如何调用张量算子。例如下面的通用尺寸计算：

```python
def output_size(height, width, short_side):
    if width >= height:
        return short_side, round(width * short_side / height)
    return round(height * short_side / width), short_side
```

它在普通 Python 中没有困难。进入导出流程后，`height` 和 `width` 可能以符号尺寸或追踪对象的形式参与计算，Python 分支、浮点运算与取整就需要被捕获、约束或特化。具体行为取决于导出器及版本。

本次记录中的阻塞点是这条 Python 形状表达，不能据此概括为 ONNX 无法表示取整。Python 内置 `round()`、Tensor 上的舍入运算和导出图中的算子，是不同层面的对象；简单替换函数名也未必能满足下游算子对尺寸参数的要求。

新式 `torch.export` 路径会处理 Python 控制流并记录形状约束，传统追踪路径也可能把样例中的决定固定下来。记录 `dynamo`、`dynamic_shapes` 或 `dynamic_axes` 等参数，有助于解释导出文件到底支持什么。

# 二、先确定交付是否真的需要动态尺寸

如果应用始终输入同一种分辨率，相应中间特征尺寸也固定，那么池化目标尺寸可以在构造适配模块时计算一次。前向只使用已确定的整数元组，不再通过 Python 从输入尺寸推导它。

这种静态适配保留的是固定输入条件下的原计算。它应沿用原有的比例、偏移和舍入规则，同时保留输入尺寸检查，让超出支持范围的输入明确失败。下面示例仅展示固定尺寸契约：

```python
import torch

class StaticPool(torch.nn.Module):
    def __init__(self, height, width, short_side):
        super().__init__()
        self.expected_hw = (height, width)
        self.target_hw = output_size(height, width, short_side)

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.expected_hw:
            raise ValueError("unexpected spatial shape")
        return torch.nn.functional.adaptive_avg_pool2d(x, self.target_hw)
```

Python 检查用于导出前或 eager 调用，导出器可能把它特化或消除，不能假定它会成为 ONNX 中的运行时检查。应用仍应根据模型输入签名校验输入。

静态特化也不保证所有池化尺寸组合都可导出。形状计算消失之后，实际池化、插值等算子仍需经过对应导出器和目标运行时验证。

# 三、静态表达要核对原来的计算

适配时最容易遗漏的是取整附近的细节。原代码若在比例表达中加入偏移，或者对长边、短边分别处理，提前计算时也必须保留。Python `round()` 的半整数处理与某些其他舍入方式不同，直接改成截断或另一个取整操作可能改变输出尺寸。

除目标尺寸外，还要核对后续 reshape、插值目标和布局。形状刚好匹配并不证明数值路径相同，尤其是把动态尺寸改成常量后，周围的分支可能一并被固定。

本次隔离适配在固定特征形状下，比较了原实现与静态表达的 FP64 前向、输入梯度和公共参数梯度，所测最大绝对差均为零；随后 ONNX Runtime 的相对 L2 误差约为 `1.70e-7`。前者检查重写是否改变该输入条件下的计算，后者检查导出与运行时转换后的输出。

梯度检查发生在 PyTorch 原实现和静态表达之间，不是验证 ONNX 的训练反向。测试中的零差异也只描述被比较的状态和输入；修改分辨率、比例或其他影响尺寸的配置后，应重新生成并核对适配结果。

# 四、导出成功之后还要确认使用范围

固定尺寸 ONNX 可以保留 `Shape` 节点，因为图中仍可能需要读取 batch 或其他维度。因此，不能按有没有 `Shape` 节点判断是否支持动态输入；同样，填写动态轴名称也不足以证明所有依赖尺寸的分支都已动态化。

应从输入签名、图内常量与尺寸约束，以及实际运行结果共同检查。若需要动态长宽，应验证长方形、长短边互换、取整临界点及不支持的尺寸，并为分支和尺寸运算选择能被捕获的表达。只在一个方形输入上通过，不覆盖这些行为。

本次交付验证范围是固定空间尺寸表达和 ORT CPU 输出核对。它解决了该形状下的导出阻塞，未将支持范围扩大为任意分辨率。更完整的中间输出与应用接口核对，见[ONNX 模型转换中的中间特征与数值对齐](ONNX_中间特征与数值对齐.md)。

# 五、参考资料

- [PyTorch ONNX：导出路径、形状约束与动态输入](https://docs.pytorch.org/docs/2.9/onnx.html)
- [Python round：舍入规则](https://docs.python.org/3/library/functions.html#round)
