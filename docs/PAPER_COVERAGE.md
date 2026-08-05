# MRAgent 论文覆盖矩阵

基准为论文 [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM
Agents](https://arxiv.org/abs/2606.06036) 及其
[官方实现](https://github.com/Ji-shuo/MRAgent)。本表记录 2026-08-05、插件
0.9.0 的状态。“结构完成”表示机制已进入核心代码和回归测试，不等于已经复现论文
全部基准分数。

| 论文机制 | 当前状态 | 本项目中的实现证据 | 仍缺什么 |
|---|---|---|---|
| Cue--Tag--Content 图 | 结构完成 | `episodes`、`semantic_memories`、`topics`、Participant 及 Cue/Tag、Participant/Aspect、Topic/Episode 边；claim 有多源表和 exact span | 图片内容仍只有附件描述符；尚无视觉内容节点 |
| Episode、Semantic、Topic 三类内容 | 结构完成 | 分层解析/持久化/索引；宿主时间；claim epistemic/conflict/quarantine/supersede/retract/stale；topic 从活动 episode 聚合 | 未实现一般化的时间有效区间推理；同义 topic 合并仍依赖模型命名 |
| 由对话构建图记忆 | 运行路径完成 | oldest-first per-message checkpoint、内容哈希、重叠上下文、后台阈值/周期 worker、失败/中断恢复和 cited-or-ignored 覆盖账本 | 需要长期 canary 验证调度积压与漂移；暂无线性扫描自动压测 |
| 群聊人物主体 | 工程扩展完成 | 平台账号主键、别名历史、mention/reply、speaker/subject 分离、重名拒绝合并、管理员别名、Bot Participant、自助删除抑制 | 不猜测跨群/跨平台现实人物；纯文本新昵称没有确定性证据时保持 unresolved |
| 查询激活状态 `Z` | 部分完成 | Harrier/BGE 初始化原论文图候选；反馈假设另有持久化激活记录、激活方法与工作图节点 | 原论文 Cue/Episode/Semantic/Topic 的每轮剪枝状态仍未全部物化 |
| 重建历史 `H` | 部分完成 | 私有 agent 保留当前轮工具消息；ledger 记录工具、参数、证据 key、结果哈希；主 Agent 的可观察 request/tool/result/response 图可持久检查 | 不持久化隐藏推理是有意的；原论文重建控制器的全部公开状态仍需统一到工作图 |
| Table 4 七类工具 | 结构完成 | 七个有类型、只读、按群约束的图遍历工具；主 LLM 默认不可见 | 工具结果预算和按路径的动态上限还可继续收紧 |
| 主动多步重建 | 运行路径完成 | 独立 Provider tool-loop；候选最低分；显式历史意图门；在线宿主证据停止门可配置 | 需要更多真实请求评估误停/漏停，并按 owner/path 继续优化步数 |
| 有证据的最终记忆摘要 | 运行路径完成 | 最终输出为 host-validated claim/source/conflict/unresolved JSON；每个结论和限定项的 source 都必须来自本轮访问；按完整结构单元截断 | 内容蕴含仍依赖模型，不等于形式证明；需扩大人工事实一致性评测 |
| 构建与重建的 token 观测 | 超出论文原型 | `experiment_runs`、`llm_usage_events`、`reconstruction_steps`；`/mrmem usage` | 运行时多轮重建目前保存 AstrBot runner 聚合值，不拆分每次内部 LLM 调用 |
| 论文消融与基准 | 部分完成 | CE/CTE/CTC 思路可由 forked run 实验；已完成首个真实调用遮罩 A/B | 尚未复跑论文完整数据集，也没有足够真实样本形成统计结论 |
| 按群物理隔离 | 工程扩展完成 | 每群独立 SQLite、事件推导 scope、SQL 二次约束、跨群测试、每群预算和自助删除 suppression | 上线前仍需 canary 验证真实 adapter 撤回与备份策略；SQLite 本身不加密 |
| 反馈驱动修订 | 行为闭环完成 | reply/词面快门、后台 proposal worker、Action--Feedback--Hypothesis 图、反向效用、阈值、双层激活、衰减/休眠和可逆合并；构建层事实已有修订状态机 | 反馈 Agent 不直接写 factual claim；管理员 proposal confirm/edit/reject UI 和更大规模误激活评测仍需完成 |

## 当前结论

当前不是“完整复现论文”的完成态，而是：

1. 论文的图结构、构建路径、候选激活、七工具和主动重建已经贯通；
2. 首个真实历史调用的严格遮罩实验已经端到端跑通；
3. 身份绑定、在线增量整理、宿主停止门、结构化 brief 和 semantic revision 已贯通；
   生产化最关键的缺口转为线上 canary、图片语义管线、原论文 active-state 统一与人工复核；
4. 论文完整 benchmark reproduction 仍应作为独立验收项，不能用一个成功案例替代。

## 下一阶段验收顺序

1. 用白名单单群做只采集/后台整理 canary，验证 backlog、撤回、Bot 输出和 24 小时成本；
2. 把原论文 Cue/Episode/Semantic/Topic 的每轮 active-state 转移统一到现有工作图；
3. 为图片建立独立的下载期限、内容 hash、OCR/视觉描述和删除策略，默认继续关闭原图保留；
4. 把反馈 proposal 的 confirm/edit/reject 接入管理员控制台；
5. 扩大真实遮罩调用样本与身份/修订测试集，再决定何时引入 ANN 或完整论文基准复现。
