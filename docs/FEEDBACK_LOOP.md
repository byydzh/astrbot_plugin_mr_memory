# 反馈图闭环

0.11.0 延续“可检查工作图而非隐藏思维链”，并把候选筛选与维护移出主请求路径。每次主 Agent
调用会建立一个按群隔离的 `interaction_trace`，只记录可观察状态：请求摘要、已激活的
前瞻假设、工具名与参数键/哈希、工具结果哈希、最终可见回复和发送组件类型。

## 运行路径

```text
主请求 -> 活动工作图 -> 主 Agent 行动/回复
                      |
后续群消息 -> reply/词面快门 -> 后台队列 -> 私有维护 Agent -> 宿主校验事务
                                      |
                 反馈节点 <- 反向效用归因 -> 前瞻假设
                           \-> 可塑语义边/动态关系类型
                                              |
下一次请求 -> 词面快门 + 私有语义门 -> 临时行为提示 -> 主 Agent
```

反馈 Agent 和主 Agent 使用不同 Provider。三个修改工具只存在于私有维护循环，主 LLM
不可见；语义激活工具同样只存在于私有重建循环。聊天正文一律作为不可信证据。

## 宿主不变量

- 数据库和每次 SQL 操作都绑定当前群 UMO；LLM 不能指定其他群。
- 维护循环只能检查和提交当前 prompt 绑定的 proposal。
- proposal 只能引用反馈发生前、配置时间窗内的 trace。
- 没有 reply-to-Bot 或反馈词面信号的普通跟聊不会形成 proposal；时间接近和同一 sender
  只会提高已有信号的分数，不能单独打开快门。
- sender 假设只能检索当前反馈发送者及群级假设；修改已有假设时还会校验 scope、时间和
  合并状态。
- 自动修改要求 `abs(feedback_valence) * confidence >= feedback_min_commit_score`；默认
  为 `0.65`，群级规则最低为 `0.8`。
- `activation_mode=always` 仅用于跨主题的通用表达偏好，且不得带 trigger；
  `activation_mode=semantic` 用于任务条件规则，必须带至少一个证据来源的 trigger。
- 词面 trigger 只负责廉价快门；没有命中字面词时，私有潜意识 Agent 可在有界候选集中
  识别真正的语义改写。发送者不匹配时不会调用 LLM。
- 反馈只改变已激活路径的 `utility`，不会倒推修改事实证据的 `confidence`。
- 可塑图修改必须等同一 proposal 先由 `mr_feedback_commit` 提交；被拒绝或未提交的反馈没有
  图写权限。
- 负向抑制/废弃只能指向目标回答 trace 中实际激活的可塑边；维护时才搜索到的替代路径
  不能为旧回答承担负向 credit。
- 写入、反馈边、证据关系和 proposal 状态在一个 SQLite 事务中提交。
- 私有维护、构建和重建共享每群滚动 token 预算；达到预算后 fail-open，不阻断主回复。

## 保留、合并与遗忘

原始证据、反馈链接和历史激活保留。TTL 或容量上限只把工作节点/假设移出活动视图，
不会删除证据。效用按半衰期衰减；低效用假设或可塑边进入 `DORMANT`。假设与可塑节点
合并均保留前身和证据，不把多个账户主体合并成一个人物。

## 成本与启用边界

`feedback_learning_enabled` 默认关闭，并且依赖 `capture_enabled=true`。开启后，只有通过
快门的后续消息进入有界后台队列，由低价私有 Provider 每次最多处理
`feedback_max_pending_per_wake` 条；维护不再同步阻塞下一次主请求。维护和重建 token 进入
按群 ledger。本阶段只在隔离实验目录与导入冒烟环境验证，尚未替换线上插件。

真实历史遮罩 A/B 的样本、评分矩阵和 token 见 [开发记录](DEVELOPMENT_LOG.md)。原始群聊
正文及逐条结果只保存在 Git ignored 的 `.dev/`。
