# MRAgent 论文覆盖矩阵

基准为论文 [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM
Agents](https://arxiv.org/abs/2606.06036) 及其
[官方实现](https://github.com/Ji-shuo/MRAgent)。本表记录 2026-08-06、插件
0.11.0 的状态。“结构完成”表示机制已进入核心代码和回归测试，不等于已经复现论文
全部基准分数。

| 论文机制 | 当前状态 | 本项目中的实现证据 | 仍缺什么 |
|---|---|---|---|
| Cue--Tag--Content 图 | 结构完成 | `episodes`、`semantic_memories`、`topics`、Participant 及 Cue/Tag、Participant/Aspect、Topic/Episode 边；claim 有多源表和 exact span | 图片内容仍只有附件描述符；尚无视觉内容节点 |
| Episode、Semantic、Topic 三类内容 | 结构完成 | 分层解析/持久化/索引；宿主时间；claim epistemic/conflict/quarantine/supersede/retract/stale；topic 从活动 episode 聚合 | 未实现一般化的时间有效区间推理；同义 topic 合并仍依赖模型命名 |
| 由对话构建图记忆 | 运行路径完成 | oldest-first per-message checkpoint、内容哈希、重叠上下文、后台阈值/周期 worker、失败/中断恢复和 cited-or-ignored 覆盖账本 | 需要长期 canary 验证调度积压与漂移；暂无线性扫描自动压测 |
| 群聊人物主体 | 工程扩展完成 | 平台账号主键、别名历史、mention/reply、speaker/subject 分离、重名拒绝合并、管理员别名、Bot Participant、自助删除抑制 | 不猜测跨群/跨平台现实人物；纯文本新昵称没有确定性证据时保持 unresolved |
| 可塑局部语义图 | 工程扩展完成 | LLM 注册/修订动态关系类型；关联边支持新增、显式认知状态修订、强化、抑制、休眠、废弃和节点可逆合并；竞争释义可并存；身份节点被宿主禁止 | 关系归一化和长期漂移仍需 canary；不能把局部语义边当作确定性身份事实 |
| 查询激活状态 `Z` | 部分完成 | embedding 只初始化候选；潜意识 LLM 每请求做语义 gate；可塑边激活、效用和有界运行状态持久化 | 原论文 Cue/Episode/Semantic/Topic 的全部逐轮剪枝状态仍未统一物化 |
| 重建历史 `H` | 部分完成 | 私有 agent 保留当前轮工具消息；ledger 记录工具、参数、证据 key、结果哈希；公开 focus、active edge 和决策状态可序列化 | 不持久化隐藏推理是有意的；原论文重建控制器的全部公开状态仍需统一到工作图 |
| Table 4 七类工具 | 结构完成 | 七个论文工具及一个可塑关联遍历扩展均为有类型、只读、按群约束；主 LLM 默认不可见 | 工具结果预算和按路径的动态上限还可继续收紧 |
| 主动多步重建 | 运行路径完成 | 独立 Provider tool-loop；embedding 候选先验；LLM 语义 gate；宿主提前停止只作可选优化 | 需要更多真实请求评估漏唤醒/无关唤醒，并继续压缩多轮工具 token |
| 有证据的最终记忆摘要 | 运行路径完成 | 最终输出为 host-validated claim/source/conflict/unresolved JSON；每个结论和限定项的 source 都必须来自本轮访问；按完整结构单元截断 | 内容蕴含仍依赖模型，不等于形式证明；需扩大人工事实一致性评测 |
| 构建与重建的 token 观测 | 超出论文原型 | `experiment_runs`、`llm_usage_events`、`reconstruction_steps`；`/mrmem usage` | 运行时多轮重建目前保存 AstrBot runner 聚合值，不拆分每次内部 LLM 调用 |
| 论文消融与基准 | 部分完成 | CE/CTE/CTC 思路可由 forked run 实验；已完成首个真实调用遮罩 A/B | 尚未复跑论文完整数据集，也没有足够真实样本形成统计结论 |
| 按群物理隔离 | 工程扩展完成 | 每群独立 SQLite、事件推导 scope、SQL 二次约束、跨群测试、每群预算和自助删除 suppression | 当前单群 canary 仍需验证真实 adapter 撤回与备份策略；SQLite 本身不加密 |
| 反馈驱动修订 | 行为与语义闭环完成 | reply/词面快门、持久任务、Action--Feedback--Hypothesis 图、可塑关系修改、实际激活路径归因、阈值、衰减/休眠和可逆合并；未提交反馈不能改图 | 不直接覆写身份或 structured claim；管理员 proposal confirm/edit/reject UI 和更大规模误激活评测仍需完成 |

## 当前结论

当前不是“完整复现论文”的完成态，而是：

1. 论文的图结构、构建路径、候选激活、七工具和主动重建已经贯通；
2. 首个真实历史调用的严格遮罩实验已经端到端跑通；
3. 身份绑定、在线增量整理、LLM 语义 gate、动态关系、结构化 brief 和反馈归因已贯通；
   生产化最关键的缺口转为线上 canary、图片语义管线、原论文 active-state 统一与人工复核；
4. 论文完整 benchmark reproduction 仍应作为独立验收项，不能用一个成功案例替代。

## 下一阶段验收顺序

1. 持续白名单单群 canary，验证 backlog、撤回、Bot 输出和 24 小时成本；
2. 压缩可塑图重建的工具上下文和缓存前缀，再把原论文各层 active-state 转移统一到工作图；
3. 为图片建立独立的下载期限、内容 hash、OCR/视觉描述和删除策略，默认继续关闭原图保留；
4. 把反馈 proposal 的 confirm/edit/reject 接入管理员控制台；
5. 扩大真实遮罩调用样本与身份/修订测试集，再决定何时引入 ANN 或完整论文基准复现。
