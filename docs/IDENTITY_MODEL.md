# 账户主体模型

MR Memory 不把昵称当作人物主键。每个群数据库中的主体由宿主字段唯一确定：

```text
Participant = current group UMO + platform_id + platform account_id
```

同一账号换群名片仍是同一个 Participant；两个账号使用相同昵称仍是两个 Participant。
插件不会自动建立跨群或跨平台的现实人物身份。

## 绑定证据优先级

1. 消息发送者的 `account_id`：宿主权威主体，同时把当时群名片记入别名历史。
2. adapter 的结构化 mention/reply：建立本条消息到目标账号的关系；其中的显示名只新增
   别名，不覆盖目标本人最近一次实际发言时的群名片。
3. 管理员 `bind_alias`：向指定账号增加高置信别名，但绝不合并两个账号。
4. 普通文本称呼：只有在当前群恰好命中一个既有别名时才可作为 claim subject 候选；
   重名或新称呼保持 unresolved。

蒸馏模型只能选择宿主提供的 `participant_key`。每条绑定人物的 claim 还必须在所引证消息
中找到 speaker、mention、reply target 或唯一别名证据。`claim speaker` 与 `claim subject`
分别保存，所以“甲说乙喜欢某物”不会默认写成甲的偏好。

## 歧义与修订

- 按重名昵称查询时，宿主只返回候选账号，不返回合并后的两人记忆；调用者必须改用
  `account_id` 或 `participant_key`。
- 单来源身份/管理员权限等高风险 claim 默认进入 `QUARANTINED`；至少需要两个独立用户
  来源才可自动进入活动事实视图。隔离期间仅供管理员审计和后续构建比对，不进入自动
  运行时事实检索；同一人重复声称不会满足独立来源阈值。
- 更正使用显式 `SUPERSEDE` 或 `RETRACT` 指向同一主体的活动 claim；旧 claim 与修订记录
  保留，但不会继续作为当前事实返回。
- Bot 是 `account_type=BOT` 的 Participant。可见回复进入普通消息真值层，并通过
  `RESPONDS_TO` 指向请求消息。

## 删除

`/mrforgetme confirm` 仅作用于发起命令的当前群账号。宿主会：

- 删除该 Participant、别名、主体 claim、相关 episode/embedding 与反馈内容；
- 擦除该账号原始消息和由其请求产生的 Bot 可见回复；
- 清除其他消息中的结构化 account/reply 引用并使相关派生图重新排队；
- 保存不可逆账号哈希作为 suppression marker，阻止之后由发送、mention、reply 或离线
  导入重新创建该主体。

其他成员自己写下的普通文本不会被字符串替换；它可能仍作为他人的原始发言存在，但不再
具有指向已删除账户的结构化绑定。
