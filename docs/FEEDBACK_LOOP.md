# 反馈图闭环

0.8.0 把“潜意识思考”实现为可检查的工作图，而不是保存模型隐藏思维链。每次主 Agent
调用会建立一个按群隔离的 `interaction_trace`，只记录可观察状态：请求摘要、已激活的
前瞻假设、工具名与参数键/哈希、工具结果哈希、最终可见回复和发送组件类型。

## 运行路径

```text
主请求 -> 活动工作图 -> 主 Agent 行动/回复
                      |
后续群消息 -> 反馈候选 -> 私有维护 Agent -> 宿主校验事务
                                      |
                 反馈节点 <- 反向效用归因 -> 前瞻假设
                                              |
下一次请求 -> 词面快门 + 私有语义门 -> 临时行为提示 -> 主 Agent
```

反馈 Agent 和主 Agent 使用不同 Provider。三个修改工具只存在于私有维护循环，主 LLM
不可见；语义激活工具同样只存在于私有重建循环。聊天正文一律作为不可信证据。

## 宿主不变量

- 数据库和每次 SQL 操作都绑定当前群 UMO；LLM 不能指定其他群。
- 维护循环只能检查和提交当前 prompt 绑定的 proposal。
- proposal 只能引用反馈发生前、配置时间窗内的 trace。
- sender 假设只能检索当前反馈发送者及群级假设；修改已有假设时还会校验 scope、时间和
  合并状态。
- 自动修改要求 `abs(feedback_valence) * confidence >= feedback_min_commit_score`；默认
  为 `0.65`，群级规则最低为 `0.8`。
- `activation_mode=always` 仅用于跨主题的通用表达偏好，且不得带 trigger；
  `activation_mode=semantic` 用于任务条件规则，必须带至少一个证据来源的 trigger。
- 词面 trigger 只负责廉价快门；没有命中字面词时，私有潜意识 Agent 可在有界候选集中
  识别真正的语义改写。发送者不匹配时不会调用 LLM。
- 反馈只改变已激活路径的 `utility`，不会倒推修改事实证据的 `confidence`。
- 写入、反馈边、证据关系和 proposal 状态在一个 SQLite 事务中提交。

## 保留、合并与遗忘

原始证据、反馈链接和历史激活保留。TTL 或容量上限只把工作节点/假设移出活动视图，
不会删除证据。效用按半衰期衰减；低效用假设进入 `DORMANT`。假设合并是可逆物化视图：
源假设标记为 `MERGED` 并指向目标，证据仍在，可执行 unmerge。

## 成本与启用边界

`feedback_learning_enabled` 默认关闭，并且依赖 `capture_enabled=true`。开启后，后续文本会
形成候选，并在下一次主请求前由低价私有 Provider 处理；每次最多处理
`feedback_max_pending_per_wake` 条。维护和重建 token 进入按群 ledger。本阶段只在隔离
实验目录运行，尚未部署到线上 AstrBot。

真实历史遮罩 A/B 的样本、评分矩阵和 token 见 [开发记录](DEVELOPMENT_LOG.md)。原始群聊
正文及逐条结果只保存在 Git ignored 的 `.dev/`。
