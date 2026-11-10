---
title: Qwen 小模型的结构化输出与任务评测
created: 2026-07-02
updated: 2026-11-10
status: ready
blog: true
tags: [Qwen3.5, SFT, 结构化输出, 多模态, 评测]
---

# Qwen 小模型的结构化输出与任务评测

小规模视觉语言模型输出一个 JSON，实际需要同时完成视觉判断、字段选择和结构生成。训练 loss 正常下降，不代表三件事都已经学会；推理端能强制生成完整 JSON，也不代表模型判断正确。

归档的一次 2B 模型评测中，未加格式约束仍出现半截 JSON；加入 `response_format: {"type":"json_object"}` 后，500 条验证样本的 JSON 解析成功率达到 100%，判断准确率约 88%。这是特定模型、验证集与推理配置下的观察，不能作为无约束生成能力或独立测试集效果的证明。下面将输出契约、约束解码和任务评测分开讨论。

# 一、先让训练目标与接口对齐

原始训练目标使用数字 `hit`，而不是字符串或布尔值：

```json
{"hit":0,"summary":"未观察到满足规则的目标"}
```

`hit=1` 表示画面证据满足任务条件，`hit=0` 表示按约定判为不满足，`summary` 给出简短的可观察依据。`0`、`"0"`、`false` 对人看起来接近，在 JSON 中却不是同一种类型。训练标签、prompt、Schema 和下游读取逻辑应使用同一份定义。

输出对齐至少包含四层：

| 层次 | 示例 | 验证方式 |
| --- | --- | --- |
| 语法 | 引号、逗号与括号完整 | 严格 JSON 解析 |
| 结构 | 字段齐全、类型正确、没有额外字段 | Schema 或等价校验 |
| 字段含义 | hit 的正负定义、summary 的作用一致 | 标注规范与接口约定 |
| 任务事实 | 判断与画面证据相符 | 独立标签和人工复核 |

其中“没有目标”和“证据不足”也要区分。看不清、输入损坏或请求失败不能自动变成 `hit=0`。如果任务允许不确定结果，应在标注和接口中显式设计相应状态；旧版二分类协议则通过独立失败状态或复核流程处理，不能事后偷偷改变 0 的含义。

对象键顺序、空格和换行通常不影响 JSON 语义。训练时使用统一的紧凑序列化可以减少无关写法，但评测应比较解析后的字段，不因键顺序不同就判失败，也不因字符串看起来像 JSON 就判成功。

# 二、JSON mode 与 Schema 约束不是同一层能力

| 方法 | 主要作用 | 没有解决的问题 |
| --- | --- | --- |
| prompt 要求“只输出 JSON” | 通过指令引导格式 | 模型仍可能输出说明、代码围栏或残缺对象 |
| JSON mode | 在支持的服务端约束 JSON 生成 | 不定义业务字段、枚举和必填关系 |
| JSON Schema 约束 | 约束对象结构与值域 | 不验证字段是否符合图像事实 |
| 应用侧校验 | 拒绝结构不符或未完整结束的输出 | 不能替代视觉任务评测 |

`json_object` 下返回 `{"answer":"不满足条件"}`，即使是有效 JSON，也不符合 `hit/summary` 接口。需要字段对齐时，应进一步定义 Schema：

```json
{
  "type": "object",
  "properties": {
    "hit": {"type": "integer", "enum": [0, 1]},
    "summary": {"type": "string", "minLength": 1, "maxLength": 160}
  },
  "required": ["hit", "summary"],
  "additionalProperties": false
}
```

`properties` 声明属性约束，`required` 要求字段存在，`additionalProperties: false` 拒绝未声明字段。只写 properties 不代表所有字段都必填。这里的 160 是示例字符上限，需要结合输出预算和实际样本调整。

Schema 的标准定义与解码引擎支持的子集也不同。嵌套对象、长度约束、组合类型等应在实际后端逐项验证，不能因为请求被接受，就假定每个关键词都已生效。JSONSchemaBench 将约束覆盖、生成效率和内容质量分别评估，这比仅报告“支持 JSON”更接近部署中的实际问题。

# 三、约束解码改变了什么

以语法约束解码为例，生成器根据已有前缀与 Schema 状态计算允许的 token 集合 $A_t$，屏蔽非法候选后重新归一化：

$$
p_{\mathrm{c}}(v\mid x,y_{<t})
=
\frac{
p_\theta(v\mid x,y_{<t})\,\mathbf{1}[v\in A_t]
}{
\sum_{u\in A_t}p_\theta(u\mid x,y_{<t})
}
$$

例如生成到 `"hit":` 后，约束可以排除不符合字段类型和值域的候选；但 0 和 1 都合法时，仍要由模型根据输入选择。语法规则能约束 token 序列，不能直接观察画面来判断哪个值为真。

约束发生在 token 层，不是逐字符补齐括号。一个 token 可能跨越多个字符，XGrammar 这类实现需要结合 tokenizer 与语法状态生成掩码。因此 Schema、模型 tokenizer 和后端版本都属于推理配置的一部分，换模型或推理引擎后需要重新验证。

以下为支持 `json_schema` 的 vLLM Chat Completions 请求参数片段，`schema` 使用上一节的对象，`messages` 包含实际图像与任务规则：

```python
request = {
    "model": served_model_name,
    "messages": messages,
    "temperature": 0,
    "max_tokens": 256,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_decision",
            "schema": schema,
        },
    },
}
```

这是后续接入 Schema 的示例，不是上述 500 条验证使用的原始配置。256 也是示例预算，需要用实际 tokenizer 和答案长度检查。格式约束并不能阻止达到输出上限、超时或连接中断；合法前缀不等于完整对象。

vLLM 还提供 `structured_outputs` 参数，旧的 `guided_json` 等字段已发生接口迁移，部署时按实际版本选用一种支持的方式。Qwen 的思考模式与 chat template 也应固定：若服务区分推理内容和最终答案，仅对最终答案字段做 JSON 校验，不能把两者拼接后再用正则截取。

# 四、SFT 学习格式，也要学习判断

回答级 SFT 的目标为：

$$
\mathcal{L}_{\mathrm{SFT}}
=-\sum_{t\in\mathrm{answer}}
\log p_\theta(y_t\mid x,y_{<t})
$$

模型会同时学习固定字段、标点、说明文字与判断值。`hit` 的关键判断只占少量 token，解释却可能很长，所以总 loss 降低可能主要来自模板和常见措辞更容易预测。需要同时观察判断指标，而不是用 loss 代替任务结果。

数据准备时应先检查：

- 模型目标是 assistant 的实际 JSON，而不是整段 messages 数组或被重复转义的 JSON 字符串。
- 同一类规则的正负边界一致，summary 与 hit 不矛盾。
- teacher 回答无法解析时单独复核，不用默认 0 填补，否则格式清洗会悄悄改变标签。
- 图像数量、顺序与模板占位符匹配，截断没有切掉答案或关键视觉输入。

小模型适合保留简短、可观察的依据。要求描述“看见了什么”，比堆叠固定结论措辞更便于复核；同时不把长推理文本当作解释质量的必要条件。

专用格式训练还可能影响通用问答。一个模型如果既用于任务判断又用于聊天，需要明确任务模板或路由，并保留非领域输入的回归集。分阶段 SFT 的具体训练取舍见 [Qwen3-VL 层次化 SFT](Qwen3-VL_层次化SFT.md)，这里不重复环境与训练配置。

# 五、应用侧仍要严格解析

接收端先检查请求是否完成、是否被拒绝、最终 content 是否存在，再解析完整对象。流式响应应先按协议拼接最终答案，不把中间片段当作最终 JSON。

下面是针对本例两个字段的最小校验器，不是通用 JSON Schema 引擎：

```python
import json

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result

def reject_constant(value):
    raise ValueError("non-JSON numeric constant")

def parse_decision(raw):
    result = json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(result, dict) or set(result) != {"hit", "summary"}:
        raise ValueError("invalid fields")
    if type(result["hit"]) is not int or result["hit"] not in (0, 1):
        raise ValueError("invalid hit")
    summary = result["summary"]
    if not isinstance(summary, str) or not 1 <= len(summary) <= 160:
        raise ValueError("invalid summary")
    return result
```

这里额外拒绝重复键、NaN/Infinity，以及布尔值或浮点写法的 hit。标准 JSON Schema 的 integer 按数值判断，可能接受 `1.0`；本例接收端进一步要求整数字面量。若采用这种更严格约定，训练样本、约束生成和验证器都应一致，不能把它误报为标准 Schema 自身的要求。

避免先把 `"false"`、`"0"` 或未知字符串强制转型：Python 中非空字符串的 bool 值为真，这类“兼容处理”可能直接翻转判断。也不要只用 `json.loads` 成功作为全部标准，它默认会接受某些非标准数字，并对重复键保留后值。

修复缺引号、截取花括号、自动补括号或重试，可以作为单独的兜底流程，但需保留原始回答与修复记录。首次生成错误不能在修复后被统计成首次格式通过，更不能将校验失败悄悄转换成任务负例。

# 六、评测同时报告格式、语义与覆盖范围

原始 500 条评测说明约束生成改善了可解析性，但“100%”只覆盖那组验证样本，且没有自动证明字段类型、重复键和证据一致性全部通过。后续应把格式指标细分，固定模型、模板、输出预算、图像处理、采样与约束后端，分别保存首轮结果和兜底后的结果。

设测试请求数为 $N$，通过完整输出检查及字段校验的数量为 $N_{\mathrm{valid}}$，其中判断正确的数量为 $N_{\mathrm{correct}}$：

$$
R_{\mathrm{valid}}
=\frac{N_{\mathrm{valid}}}{N},
\qquad
\mathrm{Acc}_{\mathrm{valid}}
=\frac{N_{\mathrm{correct}}}{N_{\mathrm{valid}}}
$$

$$
\mathrm{Acc}_{\mathrm{e2e}}
=\frac{N_{\mathrm{correct}}}{N}
=R_{\mathrm{valid}}\cdot \mathrm{Acc}_{\mathrm{valid}}
$$

这里端到端指标只统计“输出有效且 hit 正确”，解释质量另算；超时、截断和格式失败留在总分母中。若没有有效输出，条件准确率应记为不可计算，而不是填成 0 或 100%。

只在可解析样本上报准确率，容易隐藏最难样本被格式失败过滤掉的问题。反过来，把所有失败都当作负例，又会污染误报和漏报统计。建议分开报告：

| 指标 | 回答的问题 |
| --- | --- |
| 请求完成率、严格解析率、字段通过率 | 失败发生在传输、语法还是结构 |
| 有效输出上的 precision/recall/F1 | 任务判断的误报与漏报 |
| 有效覆盖率、端到端正确率 | 全部输入最终有多少得到可用且正确的判断 |
| summary 与画面、hit 的一致性 | 解释是否在描述真实且相关的证据 |
| 首轮与重试后成功率、时延、输出 token | 兜底付出了多少成本 |

summary 与 hit 相互支持，也可能一起判断错误，因此还要对照原始图像。自由文本解释不宜只做字符串精确匹配，应使用明确的证据维度或人工抽检。

至少保留四组对照：基础模型加 prompt、基础模型加 Schema、SFT 模型加 prompt、SFT 模型加 Schema。它们分别帮助判断收益来自领域学习还是格式约束。若需要比较 JSON mode，再增加单独实验，不把两种约束混成一个配置。

验证集按图像来源、事件或视频片段分组，避免同一证据的相邻帧和不同问法跨集合。除总体指标外，按正负样本、遮挡、目标尺度、证据不足与非领域输入分组。1:1 的正负评测分布不等同于实际部署分布，总准确率也不能替代低发生率任务中的误报评估。

若接口还提供连续分数，输出正确性之后还需要区分排序、运行点与概率含义，见[多模态决策模型评测：任务、阈值与概率校准](../data-evaluation/多模态决策_排序阈值与概率校准.md)。

# 七、一些反思

JSON 对齐的价值，是让错误更容易定位，而不是让错误看起来更规范。格式失败时检查模板、预算和约束实现；字段合法但判断错误时回到数据和任务边界；解释与结论冲突时检查标注与联合生成。三类问题对应的改进手段不同。

对小模型而言，有限输出预算应该优先用于正确判断和必要证据。稳定的结构可以交给约束解码与接收端校验，训练则继续承担图像理解、规则遵循和不确定性处理。把这几层分别评测，才能判断下一轮微调是否真正改善了任务能力。

# 八、参考资料

- [Qwen3.5](https://github.com/QwenLM/Qwen3.5)
- [vLLM 结构化输出与参数示例](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [JSON Schema 对象、必填字段与额外属性](https://json-schema.org/understanding-json-schema/reference/object)
- [JSON Schema 数值类型](https://json-schema.org/understanding-json-schema/reference/numeric)
- [Python JSON 解码与非标准输入处理](https://docs.python.org/3/library/json.html)
- [XGrammar 约束解码原理](https://github.com/mlc-ai/xgrammar/blob/main/docs/start/constrained_decoding.md)
- [JSONSchemaBench：约束覆盖、效率与输出质量评测](https://arxiv.org/abs/2501.10868)

