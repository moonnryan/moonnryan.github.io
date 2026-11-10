---
title: Intel B60 上的 Qwen3-VL vLLM 部署
created: 2025-12-11
updated: 2026-01-22
status: ready
blog: true
tags: [Intel, B60, vLLM, Qwen3-VL, 部署]
aliases: [/assets/tutorial/Intel_B60_vLLM部署教程.html]
---

# Intel B60 上的 Qwen3-VL vLLM 部署

Qwen3-VL-32B 的部署同时受权重、视觉编码和 KV cache 容量约束。本文整理双张 24 GiB Intel B60 上的 AWQ 部署流程，使用 Intel XPU 适配的 vLLM 镜像提供 OpenAI 兼容接口。目标是先建立可重复的启动与测试基线，再调整上下文和并发。

本流程对应归档验证的 `intel/llm-scaler-vllm:0.10.2-b6`。后续版本可能改变参数和量化执行路径，升级时应重新验证，不应直接套用旧启动脚本。

# 一、检查设备与模型

宿主机需要安装匹配 B60 的驱动，并能通过 `xpu-smi` 识别两张卡。先检查设备节点，实际编号以主机为准：

```bash
xpu-smi discovery
ls -l /dev/dri/
docker version
```

准备完整模型目录，包括配置、权重、tokenizer 和 processor。本文用 `/srv/models/Qwen3-VL-32B-Instruct-AWQ` 作为示例路径。模型来源、修订版本和校验值应一起记录；只复制权重文件不足以构成完整多模态模型。

# 二、启动隔离环境

下列节点编号是示例，需与检查结果对应。容器内的 GPU 数量应再次核验；可见卡数与 TP 数不同，会在通信初始化阶段产生额外问题。

```bash
docker run --rm -it --name xpu-vllm \
  --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 \
  --group-add "$(stat -c '%g' /dev/dri/renderD129)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD130)" \
  --shm-size=40g \
  -p 8000:8000 \
  -v /srv/models:/models:ro \
  --entrypoint /bin/bash \
  intel/llm-scaler-vllm:0.10.2-b6
```

如果镜像的驱动接口另有设备或权限要求，应按对应版本的说明补齐，并在容器内执行设备探针。原始部署使用过特权容器；这里将权限收敛为设备节点直通，实际环境还需验证该方式是否满足驱动要求。

# 三、启动模型服务

在容器内设置服务密钥，避免把固定凭据写进脚本。下面将批处理预算设为 8192，作为后续压测的起点，其余参数沿用旧镜像的兼容配置。这一预算尚未在该双卡环境复测。

```bash
read -rs -p "API key: " VLLM_API_KEY
export VLLM_API_KEY
export TORCH_LLM_ALLREDUCE=1
export VLLM_USE_V1=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name qwen-vl \
  --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --dtype float16 --quantization awq \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 \
  --max-model-len 40960 \
  --max-num-batched-tokens 8192 \
  --block-size 64
```

`max-model-len` 设置单请求的长度上限，`max-num-batched-tokens` 则决定一轮调度处理多少 token。双卡测试中，64k 上下文配置出现内存不足，随后将上限降至 40960。图片数量、输入长度和并发都会影响内存占用，测试时先用短文本确认服务能启动，再逐项增加负载。

**批处理预算与延迟取舍**

长 prompt 的 prefill 会与正在生成的请求争用计算资源。启用 chunked prefill 后，调度器把 prefill 拆成多个块，与 decode 一起调度。预算较小时，每轮 prefill 占用的时间较短，有利于缩短生成中的 token 间隔；预算增大后，长输入可以更快完成 prefill，但单轮计算和临时内存开销也会增加。

| 预算 | 调优方向 | 需要观察 |
| --- | --- | --- |
| 4096 | 较小 prefill 分块 | token 间延迟、prefill 调度次数 |
| 8192 | 更多 prefill 工作 | TTFT、总吞吐、内存峰值 |

在这套 32B 双卡配置上，可以固定输入长度、输出长度和并发，对比 4096 与 8192 下的首 token 延迟、生成间隔和总吞吐。如果长输入排队较多，重点看增大预算能否缩短等待；如果已有请求的生成变得不连贯，则需要关注 prefill 对 decode 的影响。并发上限还需结合 `max-num-seqs` 和 KV cache 容量单独调整。

**eager 模式的兼容性与性能**

这套旧镜像需要 `--enforce-eager` 绕开模型执行路径中的 Torch Dynamo 兼容问题。代价是失去相应的图编译或图重放加速：decode 每步反复执行小算子，宿主调度和 kernel 启动开销会累积，尤其容易影响低并发下的生成速度。具体影响还需对比测试；算子本身仍由 XPU 内核执行。

Intel 在 2026 年 7 月发布的 `0.21.0-b1` 中加入了实验性 XPU graph 支持，本文使用的 `0.10.2-b6` 仍保留 eager 配置。升级时，先确认当前模型和量化路径支持图模式，再检查冷启动、不同输入形状和输出正确性，最后用相同负载比较两种模式的性能。

新模型或新镜像若在首请求阶段出现量化层、dtype 或优化路径冲突，可参考 [Intel B60 的量化模型与优化路径兼容性](../engineering-runtime/Intel_B60量化兼容性.md)。其中 Qwen3.5 的记录用于说明排查方法，不作为本篇旧版 AWQ 配置的验证结果。

# 四、验证接口与输出

客户端请求中的 `model` 必须与 `served-model-name` 相同。端口可达而返回“模型不存在”时，先检查服务实际模型名：

```bash
curl --fail http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $VLLM_API_KEY"

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-vl","messages":[{"role":"user","content":"请用一句话说明什么是张量并行"}],"max_tokens":128}'
```

图文测试随后加入 `image_url` 内容，分别验证自然语言描述和固定 JSON 输出。图片文件要固定，不能依赖测试期间会变化的外部资源。每轮记录输入 token、输出 token、图片尺寸、并发、耗时和失败原因。

JSON 测试先检查请求完整结束，再严格解析并校验字段类型与值域。有效 JSON 仍可能缺少约定字段；服务支持的 JSON mode 与 Schema 约束也应按实际版本分别验证，不能把请求被接受当成约束全部生效。

保留首轮输出与重试记录，避免把超时、截断或校验失败自动记成任务负例。有效覆盖率、判断准确率及重试成本的统计方式见[多模态决策模型评测](../data-evaluation/多模态决策_排序阈值与概率校准.md)。

# 五、并发与容量验收

| 层次 | 测试内容 | 观察项 |
| --- | --- | --- |
| 单请求 | 短文本、长文本、单图 | 输出正确性、首 token 延迟 |
| 多模态 | 不同尺寸与图片数 | 视觉 token、内存增长 |
| 并发 | 逐级增加在途请求 | 成功率、尾延迟、吞吐 |
| 持续运行 | 重复请求与冷启动 | EngineCore、容器重启、OOM |

纯文本测试中，10 并发下的请求全部完成。不过，原测速脚本用字符数近似 token 数，并在收集完整流式响应后才记录首 token 时间，得到的速度和 prefill 耗时存在偏差。这里保留请求成功率，性能数据需要按下面的方法重新测量。

**测速方法**

输出 token 数优先读取服务端统计，也可以用模型对应的 tokenizer 计数。处理流式响应时，在收到第一个非空生成内容的当下记录时间，随后持续接收，直到生成结束。

| 指标 | 口径 | 主要反映 |
| --- | --- | --- |
| TTFT | 发出请求到收到首个生成 token | 排队、输入处理与 prefill |
| TPOT | 首 token 到最后 token 的时间 / 后续 token 数 | 单请求 decode 体验 |
| 总输出吞吐 | 全部完成请求的输出 token / 整轮墙钟时间 | 服务整体容量 |
| 成功率与 p95 延迟 | 同一固定测试集 | 稳定性与尾延迟 |

TTFT 包含排队、输入处理和网络传输；若要分析模型本身的 prefill 耗时，需要结合服务端计时。TPOT 只统计至少生成两个 token 的请求，未产生输出的请求单独记录失败原因。

**找出并发拐点**

固定模型、输入与输出长度，分别测试 1、2、4、8、10、16 等并发档位。每档记录单请求 TPOT、TTFT p50/p95、总吞吐、设备内存及抢占情况，并区分纯文本与图文请求。

单请求速度和总吞吐需要一起看。增加并发后，批处理通常能提高设备利用率，但每个请求的生成间隔也可能变长。当总吞吐不再明显增长、尾延迟却快速上升时，服务已经接近当前负载下的容量上限。若速度突然下降，重点检查 KV cache 不足引起的抢占与重算、prefill 与 decode 的竞争，以及双卡通信和宿主调度开销。

交互服务根据可接受的首 token 延迟和生成间隔设置并发上限，并限制入口排队长度。离线任务则更关注总吞吐，可以容忍更长的单请求耗时。

遇到 `EngineDead`，保存服务端首个异常及容器状态。客户端 500 是后果，不能仅根据它判断是显存不足。把共享内存、设备通信、量化内核和输入容量分开检查，每次只改变一个变量。

# 六、参考资料

- [Intel llm-scaler](https://github.com/intel/llm-scaler)
- [vLLM 项目](https://github.com/vllm-project/vllm)
- [vLLM 0.10.2 调度与性能调优](https://docs.vllm.ai/en/v0.10.2/configuration/optimization.html)

