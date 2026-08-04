# MRAgent 论文覆盖矩阵

基准为论文 [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM
Agents](https://arxiv.org/abs/2606.06036) 及其
[官方实现](https://github.com/Ji-shuo/MRAgent)。本表记录 2026-08-05、插件
0.8.0 的状态。“结构完成”表示机制已进入核心代码和回归测试，不等于已经复现论文
全部基准分数。

| 论文机制 | 当前状态 | 本项目中的实现证据 | 仍缺什么 |
|---|---|---|---|
| Cue--Tag--Content 图 | 结构完成 | `episodes`、`semantic_memories`、`topics` 及 Cue/Tag、Person/Aspect、Topic/Episode 边 | semantic memory 若综合多条消息，目前只保留一个主 `source_message_id`；需要多源关联表 |
| Episode、Semantic、Topic 三类内容 | 结构完成 | 蒸馏解析、来源校验、分层持久化、向量索引及端到端 fixture | 语义事实的冲突、修订和有效期尚未进入在线路径 |
| 由对话构建图记忆 | 部分完成 | `/mrmem distill`；遮罩实验支持严格截止时间和从新到旧的窗口回填 | 在线仍是管理员手动整理最近一批；缺定时/定量调度、checkpoint 和连续反向回填 |
| 查询激活状态 `Z` | 部分完成 | Harrier/BGE 初始化原论文图候选；反馈假设另有持久化激活记录、激活方法与工作图节点 | 原论文 Cue/Episode/Semantic/Topic 的每轮剪枝状态仍未全部物化 |
| 重建历史 `H` | 部分完成 | 私有 agent 保留当前轮工具消息；ledger 记录工具、参数、证据 key、结果哈希；主 Agent 的可观察 request/tool/result/response 图可持久检查 | 不持久化隐藏推理是有意的；原论文重建控制器的全部公开状态仍需统一到工作图 |
| Table 4 七类工具 | 结构完成 | 七个有类型、只读、按群约束的图遍历工具；主 LLM 默认不可见 | 工具结果预算和按路径的动态上限还可继续收紧 |
| 主动多步重建 | 结构完成但控制器未完成 | 独立 Provider 的 AstrBot tool-loop；真实调用上完成多步重建 | 纯 LLM 停止策略会过度探索；已验证的宿主证据门控尚未接入在线 runner |
| 有证据的最终记忆摘要 | 结构完成 | 潜意识只返回有界 brief，主 LLM 以临时、不可信内容接收 | 需要机器可读 claim/evidence/unresolved 格式，减少自由文本二次漂移 |
| 构建与重建的 token 观测 | 超出论文原型 | `experiment_runs`、`llm_usage_events`、`reconstruction_steps`；`/mrmem usage` | 运行时多轮重建目前保存 AstrBot runner 聚合值，不拆分每次内部 LLM 调用 |
| 论文消融与基准 | 部分完成 | CE/CTE/CTC 思路可由 forked run 实验；已完成首个真实调用遮罩 A/B | 尚未复跑论文完整数据集，也没有足够真实样本形成统计结论 |
| 按群物理隔离 | 工程扩展完成 | 每群独立 SQLite、事件推导 scope、SQL 二次约束、跨群测试 | 上线前仍需 canary 验证配置与运维路径 |
| 反馈驱动修订 | 行为闭环完成，事实修订未完成 | proposal 队列、后续反馈归因、Action--Feedback--Hypothesis 图、反向效用更新、提交阈值、双层激活、衰减/休眠和可逆合并 | 尚未让反馈直接修订 semantic fact；管理员 confirm/edit/reject UI 和更大规模误激活评测仍需完成 |

## 当前结论

当前不是“完整复现论文”的完成态，而是：

1. 论文的图结构、构建路径、候选激活、七工具和主动重建已经贯通；
2. 首个真实历史调用的严格遮罩实验已经端到端跑通；
3. 行为反馈闭环已有可检查工作图和宿主门禁；生产化最关键的缺口转为在线整理调度、
   原论文重建状态统一、宿主停止门控、semantic fact 多源 provenance 与人工复核界面；
4. 论文完整 benchmark reproduction 仍应作为独立验收项，不能用一个成功案例替代。

## 下一阶段验收顺序

1. 把已经通过消融的宿主证据门控接入运行时，并保留无门控开关用于继续 A/B；
2. 将原论文重建输出改成结构化 claim/evidence/unresolved，并把原论文 active-state 转移
   接入现有可观察工作图；
3. 增加增量整理 scheduler、反向回填 checkpoint 和幂等恢复；
4. 将 semantic memory 改为多源 provenance，再把反馈 proposal 接入管理员人工复核；
5. 扩大真实遮罩调用样本，固定人工评分 rubric，最后再做论文公开基准复现。
