# MR Memory 开发与实验记录

本文只记录可复核事实、实验配置和成本。真实群聊正文及数据库快照位于 Git 忽略的
`.dev/`，不得提交。插件 ledger 也只保存查询/结果哈希、工具参数、证据 key 和 token，
不保存模型隐藏推理。

## 2026-08-05：真实调用遮罩实验 #726

### 目标与样本

- 真实调用时间：2026-07-30 16:31:44 +08:00；
- AstrBot `provider_stats.id=726`；
- 群范围：仅记录在本地忽略的 manifest，公开开发记录不写 UMO；
- 查询类别：要求根据较早的游戏偏好发言判断一名群成员是否“口嫌体正直”；
- 真实主模型：`openai/gemini-3.5-flash`；
- 真实调用 token：20,492 input、436 output，共 20,928；
- 原回复把不存在的直接引语、游戏名和动机写成了事实。

两个源数据库的 SHA-256 已写入本地 manifest，用于验证快照一致性；公开记录不发布
私有数据库指纹。

### 遮罩协议

1. 以真实 `/chat` prompt 的时间戳 `1785400304` 为严格 cutoff；所有构建、查询和工具
   都要求 `sent_at < cutoff`。
2. 导出该群 cutoff 前最近 468 条可见消息，按 40 条一窗；窗口从新到旧构建，窗内保持
   对话时序，共 12 个窗口。
3. 每个实验使用隔离 SQLite，不能看到线上现有图，也不能在后续版本的 topic/semantic
   覆盖写入中偷看未来。
4. 图构建使用独立 DeepSeek Provider；本地 Harrier 只做向量候选初始化，不消耗 token。
5. A/B 的主模型、系统提示和最近 30 条上下文相同；唯一变量是是否附加 MR evidence brief。
6. 每个工具结果收集 source key，并再次验证其最大时间严格早于 cutoff。

物化结果：468 条 raw message、153 个图单元、485 个 embedding 文档。Harrier 的第一批
候选直接命中 semantic memory 58（score 0.520976）和 episode 55（score 0.501119）。

episode 55 的原始证据约早 38.8 小时，包含该成员对类魂游戏已经厌倦以及对其玩法、
剧情的具体评价。原话、昵称、source key 和精确时间只保存在本地审计产物。

### 消融矩阵

构建 token 是一次性背景成本，所有 fork 共用同一份 12 窗构建结果，不能在总计时重复
收费。下表的“主回答”是受控 A/B arm，不是线上原调用的 20,928 token。

| Run | 控制变化 | 重建 LLM 调用 | 工具/门控步骤 | 重建 token | Control 回答 | MR 回答 | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| r1 | 原始 LLM 自主停止 | 4 | 12 | 23,566 | 1,094 | 1,868 | 首步已拿到证据却继续浏览，最终错误返回 `NO_RELEVANT_MEMORY` |
| r2 | 加强直接证据停止提示 | 4 | 未完成 | 19,476 | — | — | 模型生成缺少必填参数的 tool call，执行器失败；保留为失败样本 |
| r3 | 无效参数可恢复 + 提示停止 | 6 | 15 | 37,316 | 1,105 | 2,121 | 找回真实证据并改善回答，但过度探索更严重 |
| r4 | r3 + 宿主直接证据门控 | 2 | 4 tools + 1 gate | 5,480 | 1,094 | 1,779 | 找回原话，明确“抢首发”无证据，严格无泄漏 |

共同的一次性构建成本：28,532 uncached input + 4,608 cached input + 20,030 output
= 53,170 token。r4 每次查询路径的受控新增成本为 5,480 重建 + 1,779 MR 主回答
= 7,259 token；这只是本样本的成本，不可直接外推平均线上费用。

r4 相比 r3 将重建 token 从 37,316 降至 5,480，下降 85.31%。门控条件为：

- 工具返回的是初始高分 episode 的原始 `query_event_context`；
- candidate score 至少 0.48；
- 查询与证据存在显著词项交集；
- 原始记录包含 source key。

门控只决定“停止浏览并进入综合”，不替模型判定命题真假。r4 最晚使用的证据时间为
`1785260988`，严格小于 cutoff `1785400304`。

### A/B 结论

Control 只能从最近上下文判断该成员当天一边吐槽一边继续玩，无法确认此前的类魂表态。
MR arm 能准确补充较早的厌倦表态和相关上下文，同时指出未检索到“抢首发/预购”的证据。
因此本例的改进不是单纯让答案更肯定，而是同时提高了 recall 和 calibration。

一个样本只能证明数据流和反事实实验方法可行，不能证明总体效果。后续必须抽取更多真实
调用，以盲评方式分别标注证据召回、事实一致性、遗漏、无根据推断和回答帮助度。

### 开发者 token ledger

0.7.0 新增三张表：

- `experiment_runs`：scope、实验类型、cutoff、查询哈希、状态和结果摘要；
- `llm_usage_events`：phase/arm/provider/model、uncached input、cached input、output、
  latency 和计量来源；
- `reconstruction_steps`：工具、参数、证据 source keys、结果 SHA-256 和延迟。

离线实验逐次记录每个 LLM call。运行时蒸馏记录单次 response usage；运行时私有 agent
直接读取 AstrBot `ToolLoopAgentRunner.stats.token_usage` 的多轮聚合值，避免只统计最终
response。管理员可在群内执行 `/mrmem usage [limit]` 查看该群最近记录。本地 embedding
只记录耗时和模型信息，不伪造 token 数。

### 本地审计产物

工作区中的主要产物：

- `.dev/experiments/masked-call-726/call.json`
- `.dev/experiments/masked-call-726/messages.jsonl`
- `.dev/experiments/masked-call-726/construction/`
- `.dev/experiments/masked-call-726/result.json`
- `.dev/experiments/masked-call-726/result_r3.json`
- `.dev/experiments/masked-call-726/result_r4.json`
- `.dev/experiments/masked-call-726/graph_r4_with_results.db`

这些文件含真实群聊证据，只在本机审计，保持 Git ignored。复现实验使用
`scripts/export_masked_call.py` 与 `scripts/masked_ab_experiment.py` 的
`construct`、`materialize`、`fork`、`evaluate --host-evidence-gate` 四阶段命令；Provider
配置必须来自本地忽略文件，不得写进命令输出或仓库。
