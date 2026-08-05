# MR Memory

面向 AstrBot 群聊的证据可追溯记忆插件原型。目标是逐步替代
AngelEye 的历史检索和 Local Reminiscence 的语义记忆。当前版本提供原始消息
真值层、LLM 图构建、embedding 候选初始化、主动遍历和离线回放能力。
0.12.1 让增量整理与真实反馈都能维护按群隔离的可塑关联图。局部梗义、反话和委婉语
可以同时保留多条竞争路径，并显式标记为 `HYPOTHESIS`、`SUPPORTED`、`CONTESTED`
或 `CONFIRMED`；embedding 只生成候选，语义状态与遍历停止由独立潜意识 LLM 判断。

研究基础：[Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036)。

> [!WARNING]
> 本插件把群聊隔离视为独立安全边界，不信任主 Agent 替它完成隔离。
> 作用域只从当前群事件的 UMO 推导，每个群使用独立 SQLite 文件，且 SQL
> 仍会校验 scope；LLM 和用户参数都不能指定其他群。
> 详细 invariant 与威胁边界见 [架构文档](docs/ARCHITECTURE.md)。

## 安全默认值

- `capture_enabled=false`：不采集任何线上消息。
- `feedback_learning_enabled=false`：反馈闭环默认关闭；开启时仍受群/发送者/时间和证据
  分数的宿主门禁约束，默认提交阈值为 `0.65`；反馈未先被宿主提交时不能修改可塑图。
- `subconscious_provider_id=deepseek/deepseek-v4-flash`：记忆推理与主 LLM 分离。
- `distillation_thinking_mode=enabled`：图构建保留模型完整思考能力；长调用采用流式接收，
  关闭思考只作为显式诊断选项，不作为省时默认值。
- `embedding_model_name=BAAI/bge-small-zh-v1.5`：插件本地运行的中文 ONNX
  embedding 模型，不经过 AstrBot Embedding Provider 或远程推理 API。
- `expose_traversal_tools=false`：主 LLM 默认看不到论文遍历工具及插件扩展工具。
- `consult_tool_enabled=true`：主 LLM 只看到一个潜意识咨询桥接工具。
- `runtime_wake_mode=every_request`：有图记忆后，每次主 LLM 请求都会尝试潜意识重建；
  可切换为 `manual_only`，仅允许主 Agent 主动咨询。
- 新消息达到 `auto_distillation_min_pending=30` 时立即整理；不足 30 条时，最迟由
  `maintenance_interval_minutes=5` 的后台轮询触发。
- 数据库固定写入本插件的 `plugin_data` 目录。
- 每个账户主体以 `platform_id + account_id` 为不可变主键；昵称只作为带时间的别名，
  重名不会自动合并。
- 每群私有 LLM 默认有 24 小时 `120000` token 预算，超限 fail-open。
- 日志默认不输出聊天正文。
- 不连接或重启 NapCat，不发送消息。

线上配置可以用 `allowed_umos` 限定群范围；留空明确表示所有群。UMO 的首段必须是
平台实例 ID，例如 `byy_official:GroupMessage:123456`，不是 `aiocqhttp` 这样的适配器类型。

## 当前结构

```text
AstrBot event adapter (main.py)
        |
        v
NormalizedMessage
        |
        v
SQLite truth store + FTS5 + layered memory graph
        |
        +-- Cue--Tag--Episode
        +-- Participant--Aspect--Structured Claim
        +-- Topic--Episode
        +-- Action--Feedback--Prospective Hypothesis
        +-- Plastic Node--Versioned Learned Relation--Plastic Node
        |
        v
embedding candidate initialization (Cue / Episode / Semantic / Topic)
        |
        v
LLM-composable bounded traversal toolkit
        |
        v
private provider tool loop (DeepSeek by default)
        |
        +-- automatic bounded memory brief before main LLM request
        +-- bounded feedback maintenance and prospective activation
        +-- persistent bounded operational state and resumable maintenance jobs
        +-- one optional consultation tool visible to the main LLM
```

`mr_memory/` 不依赖 AstrBot，可以通过普通 Python 测试和 JSONL 回放独立运行。
`main.py` 负责把 AstrBot 事件转换为核心模型。论文 Table 4 的七类工具和可塑关联、
重复媒体扩展工具只加入
插件私有 LLM 的工具循环；主 LLM 默认只能接收有长度上限的证据摘要，或调用
`mr_consult_subconscious` 请求一次更聚焦的重建。

插件使用与 AstrBot `Context.tool_loop_agent()` 相同的 `ToolLoopAgentRunner` 调用自己
配置的 provider，不继承当前会话主 LLM。插件直接持有 runner，以便记录整个多轮循环的
聚合 token，而不是误把最终 response 当成全部费用。若当前群还没有任何图记忆，自动唤醒
会直接跳过；一旦有图，每个主 LLM 请求都会运行一个有界潜意识 tick。默认不以固定向量
相似度裁决相关性，embedding 只提供候选先验；潜意识可以选中、忽略或继续遍历。最终
claim、conflict 与 unresolved 项都必须引用本轮工具实际访问的 source key，否则整份
brief 被宿主拒绝，而不是把未验证自由文本交给主 LLM。

消息按 per-message checkpoint 增量整理：批次只选择最早的 `PENDING/FAILED` 消息，附带
只读重叠上下文，成功后逐条推进；编辑或撤回会使派生图失效并重新排队。达到消息阈值或
维护周期后由有界后台 worker 池处理，不阻塞主回复。后台整理使用独立长超时，主请求的
潜意识重建仍保留较短超时。DeepSeek V4 整理默认开启完整思考并流式消费响应，32k 只作
协议安全上限，不把 4k/8k 截断当成成本控制。管理员仍可执行 `/mrmem distill`
立即处理下一批。进程若在批次中断，下一次打开数据库会把悬挂 checkpoint 恢复为有界重试。
LLM 必须为每条目标消息提供图证据或显式 ignore reason，不能静默漏掉。

本地 embedding 由 FastEmbed/ONNX Runtime 执行。默认 `BAAI/bge-small-zh-v1.5`
模型文件约 90 MB，首次真正构建向量时下载到插件自己的
`plugin_data/astrbot_plugin_mr_memory/models/fastembed` 目录，随后离线推理；默认仅用
1 个 CPU 线程和 16 条批大小，适配 2C2G 开发目标。

也可把 `embedding_backend` 设为 `sentence_transformers`，使用
`microsoft/harrier-oss-v1-270m` 等 PyTorch 模型。Harrier 查询应配置
`embedding_query_prompt_name=web_search_query`，文档向量不会加查询 prompt；
2C2G 环境建议把批大小设为 4、最大长度设为 512。模型仍完全由插件本地运行，
不会借用 AstrBot Embedding Provider。

## AstrBot 可视化控制台

插件安装后，AstrBot 的“插件页面”会自动出现“记忆控制台”。控制台与 AstrBot
仪表盘共用登录鉴权，无需开放额外端口，提供：

- 检测 AngelEye 按需产生的现有历史缓存，选择 OneBot 平台实例并按群幂等迁移；同时单独
  显示 OneBot 实际加入群数，不能把缓存群数当成完整平台群数。迁移原始消息本身不调用
  LLM，完成后才按当前整理阈值与每群 Token 上限构图；
- 直接显示实际生效范围、潜意识唤醒模式、整理触发条件、反馈窗口和 Embedding 模型；
- 所有已知群范围的消息量、图记忆量、向量量和数据库占用概览；
- Cue--Tag--Episode、Participant--Aspect--Structured Claim、Topic--Episode 图谱；
- Participant 节点、账号主键、当前群名片、别名历史和重名歧义计数；
- 在已认证插件页面绑定管理员确认的“账号 ID → 别名”，不合并账号；
- 主 Agent Action--Feedback--Prospective Hypothesis 反馈图及激活模式、效用和状态；
- 动态关系类型、可塑语义边、证据、效用和生命周期状态；
- 竞争释义的认知状态与未决说明；未决边以虚线显示；
- 点击 Episode 回溯关键词和原始聊天证据；
- 按正文或发送者检索当前群的原始消息；
- 查看待整理 checkpoint，并手动触发下一批增量整理。

控制台只使用服务端枚举的 64 位不透明 scope ID。后端会再次从对应数据库读取并
校验 UMO、平台和群号，不接受前端直接指定任意 UMO。

管理员可在群内使用 `/mrmem usage [limit]` 查看该群最近的构建与重建 token ledger。
记录按群隔离，不保存隐藏推理或完整模型输出。

账户主体的绑定优先级、歧义处理和删除语义见
[账户主体模型](docs/IDENTITY_MODEL.md)。群内还可使用 `/mrmem participants [账号或别名]`
检查解析结果，使用 `/mrmem bind_alias <账户ID> <别名>` 添加管理员确认的别名；普通成员
可使用 `/mrforgetme confirm` 删除并抑制自己在该群的后续采集。

## 离线回放

在本插件仓库根目录执行：

```powershell
python -m scripts.replay_fixture `
  --input dev/fixtures/sample_messages.jsonl `
  --database .dev/mr-memory/replay.db
```

回放是幂等的，同一 `source_key` 重复导入不会生成重复消息。

完整的离线构建与候选初始化复现：

```powershell
python -m scripts.reproduce_fixture `
  --messages dev/fixtures/sample_messages.jsonl `
  --distillation dev/fixtures/sample_distillation.json `
  --database .dev/mr-memory/reproduction.db `
  --umo shadow:GroupMessage:group-a `
  --query "最后为什么选择方案 B？"
```

离线命令使用确定性的字符 n-gram hash embedding 验证数据流；AstrBot 运行时使用
插件自管的本地 FastEmbed 模型。

## 真实调用遮罩 A/B

`scripts/export_masked_call.py` 可从本地只读快照中对齐一条 AstrBot provider 调用与
群聊历史。`scripts/masked_ab_experiment.py` 支持：

- 以真实 prompt 时间为严格 cutoff，从新到旧构建历史窗口；
- 在隔离数据库中物化图和本地 embedding，所有查询继续携带 cutoff；
- fork 同一构建结果做控制器消融，避免重复支付构建费用；
- 使用相同主模型和最近上下文运行 control/MR 两个 arm；
- 审计所有被访问 source key 的时间并逐阶段记录 token。

首个真实调用实验的方法、消融矩阵和 token 结果见
[开发记录](docs/DEVELOPMENT_LOG.md)。论文机制当前覆盖范围与缺口见
[论文覆盖矩阵](docs/PAPER_COVERAGE.md)。

## 反馈学习闭环

启用后，插件为主 Agent 的每次请求保留不含隐藏思维链的可观察工作图。后续群消息由独立
潜意识 Provider 在有界证据窗口内做归因，宿主再校验 proposal、群、发送者、时间、修改
阈值和激活模式，并在一个事务中写入反馈边与前瞻假设。下一次请求先走廉价词面门，再由
私有 Agent 处理真正的语义改写；主 LLM 只接收临时、有限的行为提示。

反馈会反向调整实际激活路径的效用，但不会把情绪反馈伪装成事实置信度。可塑图修改必须在
对应反馈 proposal 已通过宿主提交之后发生；负反馈只能抑制确实参与过目标回答的边，不能让
维护时才发现的正确路径替旧回答背锅。TTL、遗忘和合并只改变活动视图，原始证据保持可追溯。完整设计与配置边界见
[反馈图闭环](docs/FEEDBACK_LOOP.md)。

## 测试

```powershell
python -m unittest discover -s tests -v
```

真实群聊 embedding 选型目前使用证据式模型初标集，构建规则、隐私边界和评测口径见
[群聊检索测试集](docs/RETRIEVAL_BENCHMARK.md)。真实正文只保存在 Git 忽略的
`.dev/`，不会进入仓库或外部推理服务。

## 当前边界

- 图片/文件只保留类型、名称和引用 SHA-256；每群最多维护 512 个重复媒体聚合项，
  只含次数、独立发送者数和最多 8 个 source key。不会下载图片、保存媒体字节、OCR、
  生成视觉描述、建立视觉向量或默认调用多模态模型；哈希本身不作为图意证据。
- SQLite 仍是明文；Linux 下数据库与 WAL/SHM 会尽力设为 `0600`，备份加密仍由部署层负责。
- Participant 只在单群内绑定平台账户；不会猜测跨群、跨平台账号属于同一现实人物。
- 纯文本出现的新称呼若没有 speaker、结构化 mention/reply、唯一旧别名或管理员绑定，
  会保持 unresolved，不会强行连接。
- semantic claim 已支持多源 evidence、冲突、quarantine、supersede/retract；增量整理与
  反馈 Agent 都可以维护独立的可塑关联图，但不直接覆写账户身份或结构化事实。
  重复 upsert 不会覆盖边的认知状态，必须通过带证据的 `revise_edge` 显式修订。
  `QUARANTINED` 不进入
  自动事实检索，达到独立来源阈值后才晋升，事实修订仍由构建证据触发。
- 自动反馈提交已有词面快门、后台队列、阈值与审计；完整管理员
  confirm/edit/reject proposal 控制台仍是后续项。
- 线上生效范围由 `allowed_umos` 明确控制；真实历史遮罩 A/B 与运行时 token ledger
  均保留用于复核。
