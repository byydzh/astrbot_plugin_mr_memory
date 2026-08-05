# MR Memory 开发与实验记录

本文只记录可复核事实、实验配置和成本。真实群聊正文及数据库快照位于 Git 忽略的
`.dev/`，不得提交。插件 ledger 也只保存查询/结果哈希、工具参数、证据 key 和 token，
不保存模型隐藏推理。

## 2026-08-06：未决语义与重复媒体锚点 / 0.11.0 单群 canary

可塑关系现在持久化 `HYPOTHESIS`、`SUPPORTED`、`CONTESTED`、`CONFIRMED`
四种认知状态及未决说明。增量整理可以直接建立竞争释义，后续证据必须通过带来源审计的
`revise_edge` 显式修订；重复 upsert 不得静默覆盖已有认知状态。控制台以虚线显示未决/
竞争路径，并在节点检查器中展示疑虑。

重复图片与媒体只建立每群有界的 opaque-hash 聚合：最多 512 项、每项最多 8 个 source key，
不保存媒体字节、URL、OCR、caption 或视觉向量，也禁止潜意识从哈希和频率猜测图意。它只
提供“某个不可见媒体反复出现”的上下文锚点，真正释义仍须由可引用的群聊反馈建立。

普通设置页缩减为 7 个运行项；兼容性与实验调参仍保留在 schema 中但默认隐藏。完整回归
64 项通过，另通过 `compileall`、配置 JSON、浏览器脚本语法和 diff whitespace 检查。
服务器使用 AstrBot 4.27.1 加载插件 0.11.0，单群白名单 canary 已启用：独立 DeepSeek
Provider、Harrier 本地 embedding、增量整理、反馈维护、主请求唤醒和高层咨询入口开启，
底层遍历工具不向主 LLM 暴露。6185/6199 与 OneBot v11 适配器验证正常，NapCat 未重启；
canary 启动时尚无白名单群新消息，因此没有误建其他群的 scope 数据库。

## 2026-08-05：truth layer v2 / 0.9.0 收口验证

本轮针对研究原型审查中的线上风险统一实现了增量 checkpoint、宿主停止门、账户主体、
Bot 输出、reply/mention、结构化 claim、多源 provenance、事实修订、后台维护和成本预算。
身份层采用 `UMO + platform_id + account_id`，昵称只作带时间别名；重名查询必须返回候选
账号并停止人物记忆读取，不能把两人的 claim 合并后再附一个歧义标记。

确定性身份/修订矩阵：

| 场景 | 宿主行为 | 回归结果 |
|---|---|---|
| 同账号换昵称 | 保持一个 Participant，累计别名历史 | PASS |
| 两账号同昵称 | 保持两个 Participant，昵称查询拒绝合并 claim | PASS |
| 他人用 mention 说目标偏好 | speaker 与 subject 分离，绑定 mention account ID | PASS |
| mention/reply 携带玩笑昵称 | 只新增低权别名，不覆盖本人最近实际群名片 | PASS |
| 旧消息晚于新消息补录 | 保留较新实际群名片，仍补全历史别名 | PASS |
| 单来源“我是管理员” | 进入 `QUARANTINED` 且不检索；同人重复无效，第二独立来源才晋升 | PASS |
| 明确更正旧偏好 | 新 claim `ACTIVE`，旧 claim `SUPERSEDED` 且有 revision | PASS |
| 自助删除后再次 mention/导入 | 擦除结构化引用并由 suppression hash 阻止重建主体 | PASS |
| 构建进程在 `PROCESSING` 中断 | 下次打开恢复为 `FAILED` 并从相同 source/hash 有界重试 | PASS |

从真实遮罩实验的 v7 图数据库复制件执行 v8 迁移：468 条消息全部建立
`message_processing`，识别 11 个 Participant，schema 升至 8，原文件未修改。插件源码还在
服务器 `/tmp` 使用线上 AstrBot 4.27.1 的 Python 3.13 解释器完成导入和 schema 冒烟；
线上进程 PID 未改变，没有替换插件、修改配置或重启 AstrBot/NapCat。默认依赖和可选
Harrier 依赖已拆分，CI 新增 AstrBot 4.27.1 import contract。

核心回归目前 53 项，另执行 `compileall`、配置 JSON、浏览器脚本语法和 `git diff --check`。
这证明迁移与宿主不变量可执行，不等于线上 canary 或新一轮总体效果 A/B；真实效果结论仍
以以下遮罩实验为限。

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

## 2026-08-05：后续反馈闭环真实历史 A/B

### 实现对象

0.8.0 为每次主请求持久化不含隐藏思维链的可观察工作图：request、已激活假设、主 Agent
工具名与参数键/哈希、工具结果哈希、可见 response 和发送组件类型。后续消息进入反馈
proposal，由独立 `deepseek/deepseek-v4-flash` 维护 Agent 检查有界上下文，再由宿主校验
proposal 绑定、严格时间顺序、群/发送者 scope、修改阈值和激活模式，最后在一个 SQLite
事务中写入 Feedback--Hypothesis 边。

负反馈只对实际激活路径做 signed utility credit，不修改事实置信度。前瞻假设分为：

- `always`：跨话题表达偏好，trigger 必须为空，直接走本地快门；
- `semantic`：任务条件规则，必须有 trigger；字面未命中时交给私有潜意识 Agent 在有界
  候选内判断语义改写。

完整不变量见 [反馈图闭环](FEEDBACK_LOOP.md)。

### 样本与遮罩协议

从本地只读真实群聊快照选取两条明确的人类纠正及其更晚真实调用：

1. `no_followup`：用户要求不要再用“你要是想”征询是否继续，而应直接给后续内容；
2. `forced_choice`：用户否定端水回答，要求二选一时只能选一个。

公开记录不写 UMO、数字 ID、昵称或完整上下文。每个 target 只包含其 cutoff 前最近 12 条
消息；feedback 本身也必须严格早于 target。control 与 memory 使用相同的历史 persona、
最近上下文和当前线上默认主模型 `openai/gemini-3.5-flash`，每个 arm 运行 3 次并交替顺序。
维护/激活只执行一次，最终回答复用同一已验证学习结果。主回答上限固定为 1200 token，
避免 500-token 调试上限被模型内部推理耗尽而产生空串或截断。

确定性评分口径：`no_followup` 要求产生非空回答且不再出现征询是否继续的句式；
`forced_choice` 要求明确选择西餐或日料之一，并且不出现条件式端水、反问或再次把选择交还
用户。该口径只检查被反馈指出的行为，不评估答案中额外营养叙述的事实性。

### 最终矩阵

| 样本 | 历史真实回复 | 当前模型 Control | MR Feedback | 负对照误激活 |
|---|---:|---:|---:|---:|
| no_followup | FAIL | 3/3 | 3/3 | 0 |
| forced_choice | FAIL | 0/3 | 3/3 | 0 |

`no_followup` 被学习为 sender-scoped `always` 规则，目标调用经零 token 本地快门激活；另一
发送者的无关二选一负对照在 scope gate 被直接拒绝。当前 Gemini 已在 control 中自行避免
旧问题，因此此样本是天花板结果，不能声称记忆带来增益。

`forced_choice` 被学习为 sender-scoped `semantic` 规则。词面 trigger“谁才是/只能选一个”
没有出现在 target“西餐还是日料”中，私有激活 Agent 仍以 0.85 relevance 建立语义连接；
同一发送者的无关改名请求返回 `NO_APPLICABLE_FEEDBACK`。Control 三次均重新给出条件式
两边建议，Memory 三次均直接选择日料，形成该样本上的 0/3 -> 3/3 行为提升。

### Token ledger

| 样本 | 一次性维护 | 目标激活 | 负对照激活 | 3 次 Control | 3 次 Memory |
|---|---:|---:|---:|---:|---:|
| no_followup | 5,361 | 0 | 0 | 4,358 | 4,991 |
| forced_choice | 5,138 | 979 | 832 | 3,459 | 4,187 |

两个样本的一次性反馈维护与激活合计 12,310 token；最终 12 个回答 arm 合计 16,995
token。这里报告 Provider usage 原值，不用字符数估算。真实运行时每个 feedback proposal
都可能产生维护成本，因此功能保持默认关闭，并由每次唤醒上限和提交阈值约束。

### 调试发现与边界

- 首轮维护 prompt 没有给出完整 decision schema，Agent 无法可靠提交；已补成精确 JSON
  schema。
- Provider 曾返回截断的 tool-call JSON；harness 现在与运行时一样把工具错误回传并允许
  重试，而不是把失败当实验结果。
- 1000-token 维护调试上限出现“完成检查但没有提交”；最终维护上限提高到 1800。
- Agent 曾把跨话题的“不要征询是否继续”误分为 semantic；prompt 现明确要求按未来规则
  的适用范围分类，而不是按产生反馈时的话题词分类。
- 两个样本只证明端到端反馈、语义桥接和负门控可以工作，不构成总体效果统计。下一阶段
  需要更多不同用户/任务的盲评，并单独加入事实一致性与过度服从指标。

本地审计产物为 `.dev/experiments/feedback-ab/bundle.json`、`result.json` 和 `report.md`；
逐条真实正文、Provider 输出和失败前版本均保持 Git ignored。

## 2026-08-05：可塑局部语义图与双案例真实历史 A/B / 0.10.0

本轮把 embedding 从最终相关性门槛降为候选生成先验，由独立潜意识 LLM 决定是否相关、
沿哪种关系继续遍历以及何时停止。新增的可塑图只接受 `concept`、`behavior`、`symbol`、
`topic`、`preference`、`procedure` 节点；账户身份、作者、reply、撤回和原始证据仍由宿主
真值层管理。关系类型可由模型注册并带版本修订，边可新增、强化、抑制、休眠、废弃或合并，
每次修改必须引用当前群已检查的 source key。

实验把最大群的 29,799 条历史消息导入一次性隔离数据库；提交记录不保留真实群号，并严格按历史时间切片
维护局部语义图，并用目标调用 cutoff 阻止未来消息泄漏。潜意识使用 `deepseek-v4-flash`，表层回答使用 `gemini-3.5-flash`；线上数据库、
插件目录和 AstrBot 进程均未修改。评分由同一盲评 rubric 产生，满分 5：

| 案例 | Arm | 评分 | 识别群内语义 | 外部同名幻觉 |
|---|---|---:|---|---|
| 好女孩 | A0 无群记忆 | 1 | 否 | 否 |
| 好女孩 | B1 / P1 | 4 | 是 | 否 |
| 好女孩 | B2 / P2 | 4 | 是 | 否 |
| 好女孩 | B3 / P3 | 5 | 是 | 否 |
| 好女孩 | B4 / P4 | 5 | 是 | 否 |
| 阿拉蕾 | A0 线上真实旧回答 | 0 | 否 | 是 |
| 阿拉蕾 | B1 查询前图记忆 | 4 | 是 | 否 |
| 阿拉蕾 | B2 负反馈维护后 | 4 | 是 | 否 |

“好女孩”最终重建为三个可组合用法：MyGO 表情包梗、对 Bot 是否听话的戏谑标签、以及
随语境变化的万能鉴定/调侃语。“阿拉蕾”在反馈前已经从群史正确解析为《梦限大》角色，
不再落到《阿拉蕾》或数码宝贝等外部同名词；负反馈后的答案分数没有继续上升，原因是两个
MR arm 都已解决名词消歧，剩余扣分来自第三条性格概括略有外推。

反馈复验同时发现一处归因错误：维护 Agent 曾试图抑制查询后才发现的正确“阿拉蕾”边，
但真实旧回答根本没有激活 MR。宿主现只允许负反馈修改目标回答 trace 中实际激活的边；
8 条提案提交 7 条、拒绝该 1 条误伤，正确主路径效用由 `1.3` 增至 `1.8`。此外，可塑图
修改必须等对应反馈 proposal 先达到 `COMMITTED`，避免反馈被拒绝但图已写入的半提交。

成功实验共 37 次模型调用、221,659 token：潜意识 28 次 / 209,490 token，表层 9 次 /
12,169 token；其中普通输入 113,175、缓存输入 85,632、输出 22,852。调试另有一次未能
持久计费的中止、4,397 token 的状态哈希失败和 95,717 token 的证据引用失败；后两者促成
了状态校验与工具结果去重修复。该数字是多阶段研究压力测试成本，不代表线上每条消息成本；
上线前需缓存稳定图前缀、缩短工具结果并减少无必要的重复重建。

本地完整产物位于 `.dev/benchmarks/plastic_semantics_v1/`：`report.html`、`REPORT.md`、
`result.json` 和逐调用 usage ledger。真实正文和模型逐条输出均保持 Git ignored。
