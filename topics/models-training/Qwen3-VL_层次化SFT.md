---
title: Qwen3-VL 层次化 SFT 的策略与边界
created: 2026-03-06
updated: 2026-03-20
status: ready
blog: true
tags: [Qwen3-VL, SFT, Ascend, 微调, 过拟合]
---

# Qwen3-VL 层次化 SFT 的策略与边界

Qwen3-VL 的领域微调同时涉及输出格式、任务判断和视觉特征适配。三阶段方案先限制更新范围，再按验证结果扩大可训练模块，目的是减少小规模领域数据对原有能力的影响。冻结视觉参数能保留其权重，却不能保证整模型的通用问答不变，语言端的更新同样会改变回答习惯。

这里的“层次化”指逐步调整训练范围，监督仍然是教师生成并经复核的回答。没有额外的特征、概率分布或 logits 损失，不能把三个阶段分别称为特征蒸馏或分布蒸馏。三个阶段使用同一类图文任务时，格式和判断也会同时学习，阶段名称只描述侧重点。

# 一、三个阶段的作用

| 阶段 | 设计中的更新范围 | 主要目标 | 验收内容 |
| --- | --- | --- | --- |
| 格式与指令 | 冻结视觉塔，更新语言端部分层 | 学会任务字段和回答格式 | JSON 与指令遵循 |
| 领域适配 | 语言端全量，视觉末段及指定连接模块 | 学习领域视觉与判定 | 分场景正负样本 |
| 联合微调 | 视验证结果全量解冻，降低学习率 | 检查联合更新是否还有收益 | 独立测试与通用能力 |

Qwen3-VL-2B-Instruct 的官方配置有 28 个语言层和 24 个视觉 block，因而“语言后 10 层”对应索引 18–27，“视觉后 4 层”对应 20–23。这是该配置的划分，不是所有 Qwen3-VL 规模都应照搬的数字。

语言高层更接近输出，只能作为限制更新范围的理由，不能认为这 10 层专门负责 JSON。类似地，把视觉前 20 层统一视为边缘、颜色，后 4 层统一视为行为语义，也不足以解释模型中各层的实际作用。

# 二、冻结清单必须与框架实现一致

LLaMA-Factory 中的 `freeze` 是部分参数微调模式，不表示其余所有名字带 freeze 的参数都用于禁止训练。核对官方参数定义与 adapter 实现后，需要区分：

| 参数或模式 | 实际含义 |
| --- | --- |
| `finetuning_type=freeze` | 按配置选择部分可训练层和模块 |
| `freeze_trainable_layers=10` | freeze 模式下选择语言端最后 10 层；仍受模块过滤影响 |
| `freeze_extra_modules` | freeze 模式下额外设为可训练的非隐藏层模块，不是冻结黑名单 |
| `finetuning_type=full` | 走全参数分支，再应用视觉塔、投影层等禁止训练规则 |
| `freeze_vision_tower` / `freeze_multi_modal_projector` | 依据该模型的模块映射冻结对应部分 |

因此，`full` 配合 `freeze_extra_modules=visual.blocks.0,...` 不能实现“冻结视觉前 20 层”。核对的官方 full 分支不读取这份列表，而在 freeze 分支中它的含义又是额外开放训练。若训练环境没有自定义改动，视觉塔与投影层的冻结开关都设为 False 时，实际更新范围可能比设计大得多。

第二阶段的视觉末 4 层方案，需要在框架确定训练模式之后、构造优化器之前，按实际模块名设置参数，并核对优化器参数组。现有材料没有完整保存当时的代码提交与逐模块训练清单，不能仅根据启动命令确认历史训练已经满足该方案。

Qwen3-VL 还不只有一个末端投影层。2B 配置的 DeepStack 从视觉 block 5、11、17 提取特征，经各自的 merger 送入语言端。如果只开放 block 20–23，前面三个分支的特征提取仍被冻结，其 merger 是否更新要另行指定。主 merger、DeepStack merger、patch embedding 与位置嵌入，都应与视觉 block 分开检查。

下面的检查应放在框架应用冻结规则后，用实际图文 batch 产生的 loss 反传：

```python
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
print("trainable params:", sum(p.numel() for _, p in trainable))

loss.backward()
with_grad = [n for n, p in trainable if p.grad is not None]
print("parameters with gradients:", len(with_grad))
```

总数之外还要保存各模块名称，确认冻结块没有梯度、计划更新的模块进入了优化器。梯度存在也不等于参数必然更新，必要时比较 optimizer step 前后的参数变化。若先构造优化器再解冻新参数，还需重建或补充参数组。

若同时改写视觉 patch embedding，还需按实际冻结范围验证前向与反向；保留原参数的实现见 [视觉 patch embedding 的等价改写：从 Conv3D 到 Linear](../graphs-operators/AMD_Conv3D等价改写.md)。

# 三、阶段切换需要重新确认训练状态

从上一阶段目录加载模型，是以其权重开始一轮新训练，不等于恢复上一阶段的优化器、调度器和步数。改变可训练范围后，应明确为新阶段构造优化器并设置 warmup；不能把它与同一任务中断后的 `resume_from_checkpoint` 混为一谈。

独立 LoRA 冒烟实验也不是 freeze 阶段的同一种产物。若阶段输出只有 adapter，后续 full 训练需要连同基础模型正确加载或先完成合并，不能把 adapter 目录直接当作完整模型。LoRA target 名称覆盖了哪些语言和视觉模块，也要看实际匹配结果，不能由 `q_proj` 等名字推断覆盖了整个多模态模型。

初始笔记示例按三阶段使用 $10^{-5}$、$5\times10^{-6}$、$10^{-6}$，对应 3、5、3 个 epoch。这些是实验配置，不是已证明的最优学习率或必要轮数。后续记录收紧到 1、2、1 个 epoch，并下调到 $3\times10^{-6}$、$8\times10^{-7}$、$3\times10^{-7}$，反映的是减少领域数据重复更新的尝试。

按每设备 batch 4、梯度累积 8、两路数据并行计算，一次优化器更新名义上累计 $4\times8\times2=64$ 条样本。但可见两张 NPU 不代表启动了两路训练，应核对实际 world size。多图数量、分辨率和文本长度不同，也会使相同样本 batch 对应不同的 token 预算。BF16 则是这套设备环境采用的精度配置，不能据此断言所有 NPU 任务都以它最优。

adapter 进入编译型端侧运行时后，还涉及预留分支、rank 和加载后的缓存状态。一个具体实现见 [Sophgo LoRA 编译预留与动态加载](Sophgo_LoRA部署.md)。

# 四、图文输入与验证配置要一起固定

`image_max_pixels=65536` 限制的是每张图的像素预算，不是固定边长，也不是每条多图样本的总预算。这个量级相当于 $256^2$ 像素；在小目标任务上，应先确认缩放后还有足够视觉证据，而不是通过解冻更多层补偿已经丢失的细节。

多图样本要核对媒体数量、顺序、占位符与 processor 输出，检查实际视觉网格和截断后的监督 token。`cutoff_len=4096` 是否足够，需要结合处理后的序列确认，不能只检查原始 JSON 的字符数。

数据注册名必须与 `dataset_info.json` 的键一致，媒体相对路径也需要在训练进程中可解析。只把 train.json 复制到另一个目录，并不会自动带走图片。训练与推理使用匹配的多模态模板、processor、思考模式和停止条件，再比较同一组生成结果；不要只更换模板名字来弱化 JSON 习惯，却没有检查实际序列。

指定 `eval_dataset` 不等于已经按期运行验证。还要明确配置评估策略、频率、保存策略和模型选择指标，并检查是否产生 eval 日志。训练 loss 曲线不能替代验证曲线，教师强制下的 eval loss 也不能替代实际生成测试。

# 五、训练下降与效果提升

归档曲线中，训练 loss 从第一阶段约 2.5 降到 0.5，第二阶段进一步降至约 0.30，第三阶段在约 0.28–0.34 波动。这说明模型更容易预测训练答案，却不能证明每一阶段分别改善了格式、视觉判断和鲁棒性。固定字段、JSON 标点与说明文字也参与 loss，领域判定只是其中一部分。

| 评测项 | 应观察的内容 |
| --- | --- |
| 格式正确率 | 可解析、字段齐全、类型合法 |
| 任务判断 | 每场景准确率、正例召回、负例误报 |
| 难例 | 遮挡、小目标、边界条件 |
| 泛化 | 新来源、未见视频、不同问法 |
| 通用能力 | 未参与领域训练的基本任务 |

后续部署测试出现过重复回答、普通问答被强制套成 JSON 等现象。降低训练预算并弱化任务提示后，通用问答有所恢复，但领域数据效果下降，最终仍选择已有领域模型用于专门任务。这不是一次“全量微调后所有能力同时提升”的结果，而是领域专用与通用回答之间的取舍。

这些观察主要来自部署测试，还需用相同输入分别比较浮点 checkpoint 与量化产物，排除模板和部署链路的影响，不能把所有异常都归因于学习率。若目标本来就是专门任务，可以明确限定模型用途；若还要承担通用问答，则需要独立回归集，并验证混合数据或较小更新范围是否有效。

全量 SFT 也不等于量化感知训练。没有模拟量化或相应损失时，第三阶段不能被称为“为 INT4 提供最稳定模型”。量化后的格式、判断与通用能力仍要对同一浮点基线重新评测，并将输出失败保留在端到端指标的分母中。相关统计见[多模态决策模型评测](../data-evaluation/多模态决策_排序阈值与概率校准.md)。

# 六、一些反思

分阶段方案首先要证明训练范围确实按设计变化，再讨论这种变化有没有收益。阶段命名、参数名称与 loss 下降都不能代替这两项验证。

更好的基线包括同预算的单阶段 LoRA、冻结视觉的语言端微调和分阶段方案。没有这些对照，适合把三阶段写成已执行策略，而不是已证明最优的方法。训练流程中的设计意图与实验结论，应分别保存。

# 七、参考资料

- [Qwen3-VL-2B-Instruct 模型配置](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json)
- [LLaMA-Factory 微调参数定义](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/finetuning_args.py)
- [LLaMA-Factory full / freeze / LoRA 实现](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/adapter.py)
- [LLaMA-Factory 多模态冻结与模块映射](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/visual.py)
- [Qwen3-VL DeepStack 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Transformers 训练与验证参数](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)

