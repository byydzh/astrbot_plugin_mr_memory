# MR Memory

面向 AstrBot 群聊的证据可追溯记忆插件原型。目标是逐步替代
AngelEye 的历史检索和 Local Reminiscence 的语义记忆。当前实现提供原始消息
真值层、后台 LLM 图构建、本地 embedding 候选初始化、来源审计和离线回放能力。
回答时每次请求先固定 `RequestSnapshot`，只在本群 SQLite 中并行检索别名观察、FTS5、
本地 embedding、情节和可塑图，再编译为全局来源有界、原文不重复的 EvidenceAtomPack。
插件配置的 resident Reader 进行一次人物指代与记忆语义的联合推理，宿主严格校验后才把
最多 12000 字符的语义简报交给 AstrBot 主模型。人物、作品、实体和主题保留独立指称，
说话者与陈述对象分别记录；原始摘录、来源键与内部审计证书不进入主模型提示。
在线没有修复调用、provider 切换或
自动深挖；局部梗义、反话和委婉语可以保留多条竞争路径，并显式
标记为 `HYPOTHESIS`、`SUPPORTED`、`CONTESTED` 或 `CONFIRMED`。embedding 只排序候选，
稳定账号、作用域、cutoff 和来源仍由宿主约束。

连续对话最多保留六个已验证来源的指针；下一轮按新的时间截止重新读取原文，不把上一轮
模型生成的回答保存为事实。选中的持久记忆先核验其完整来源关系，再截取有界原文用于
Reader；缺失、删除、跨群或越过截止时间的来源会使该条派生记忆不可用。Reader 返回后
重新核验实际使用的记忆与来源，无关后台写入不会使它们失效。机器人、人类及未知作者的
来源角色贯穿简报，避免把历史机器人发言提升为人类证词。

研究基础：[Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036)。

可发布的测试语料仅限合成数据。真实聊天、个人查询、人物称呼、群账号和人工验收答案仅放在
Git 忽略的私有目录；运行日志和含原文的报告也不提交。工作树清理不会自动抹除旧提交或外部副本。

> [!WARNING]
> 本插件把群聊隔离视为独立安全边界，不信任主 Agent 替它完成隔离。
> 作用域只从当前群事件的 UMO 推导，每个群使用独立 SQLite 文件，且 SQL
> 仍会校验 scope；LLM 和用户参数都不能指定其他群。
> 详细 invariant 与威胁边界见 [架构文档](docs/ARCHITECTURE.md)。

## 安全默认值

- `capture_enabled=false`：不采集任何线上消息。
- `feedback_learning_enabled=false`：反馈闭环默认关闭；开启时仍受群/发送者/时间和证据
  分数的宿主门禁约束，默认提交阈值为 `0.65`；反馈未先被宿主提交时不能修改可塑图。
- `local_serving_enabled=true`：回答前本地检索后调用一次插件 resident Reader；失败时不注入。
- `local_serving_timeout_seconds=180` 只用于异常卡死保护（可设 1-600 秒），不再把
  正常慢检索当成缺失记忆；运行账本分别记录 runtime/service readiness、interaction trace、
  snapshot、混合词法召回、其余 retrieval、Reader TTFB/总时长、双重 source audit、compile
  和账本写入的实际耗时。
  `local_serving_max_chars=12000` 只限制最终注入主模型的语义简报；Reader 输入由
  `local_serving_max_items=12` 的全局来源上限、来源去重和分层边际选择控制，不设置会把正常
  证据直接砍掉的 12k prompt 硬门槛。来源上限不等于字符上限，长消息及 compaction 之前的
  向量、episode 与人物扩展成本会分别记账。本地读取超时会明确记为错误且不注入。
- 在线检索不再用插件级全局锁串行所有请求；本地 embedding 按公平队列执行，后台文档
  索引会在批次间释放推理槽。普通聊天跳过仅供明确人物枚举使用的语义身份全表扫描，
  引用来源使用一次批量 fail-closed 审计。
- `subconscious_provider_id=deepseek/deepseek-v4-flash`：默认供一次在线 resident Reader 以及
  后台消息整理、反馈维护使用；不继承或替换 AstrBot 当前会话的主模型。
- `local_serving_reader_provider_id=""`：可单独选择 AstrBot 已配置的 Reader provider。
  默认留空沿用 `subconscious_provider_id`；显式选择只影响回答前 Reader，后台整理与
  反馈仍使用原 provider。调用账本与证书模型版本记录实际 Reader；所选 provider 不可用
  时记录失败，不会自动切换。
- `distillation_thinking_mode=enabled`：图构建保留模型完整思考能力；长调用采用流式接收，
  该设置只作用于后台消息整理。
- `feedback_thinking_mode=enabled`：反馈判读独立配置思考模式；切换模式后仍需通过原有
  归因、来源和提交阈值校验。有正文或合法 JSON 不等于正确学习了反馈。
- `local_serving_reader_thinking_mode=disabled`：在线 Reader 默认直接生成严格 JSON，避免
  隐藏推理耗尽输出预算。它仍是同一 provider 的一次调用；没有 repair、retry 或模型切换。
- `embedding_model_name=BAAI/bge-small-zh-v1.5`：插件本地运行的中文 ONNX
  embedding 模型，不经过 AstrBot Embedding Provider 或远程推理 API。
- 主 LLM 始终看不到 MR Memory 的遍历、反馈或咨询工具；旧配置也不能重新开启。
- `consult_tool_enabled=false`：查询时远程深挖工具固定不暴露给主模型。
- 新消息达到 `auto_distillation_min_pending=150` 时立即整理；不足 150 条时，最迟由
  `maintenance_interval_minutes=1440`（一天）的后台轮询触发。
- 后台新消息整理使用 `private_daily_token_budget`，反馈学习使用独立的
  `feedback_daily_token_budget`；回答前 resident Reader 的一次调用单独记录 usage，
  没有账单证据时 cost 明确为 `UNKNOWN_NO_BILLING_EVIDENCE`。
  外部运维工具一次性写入的旧历史只保留独立审计账本，
  不设伪装成日常策略的每日额度，也不会挤占这两类在线预算。
- 数据库固定写入本插件的 `plugin_data` 目录。
- 每个整理批次在同一数据库事务中提交记忆图、来源覆盖与进度，再刷新 embedding 索引。
  索引失败明确记录，不回滚已提交事实，也不重复提取该批原文；目前没有独立的自动索引
  修复任务，出现此类失败时仍需定向修复索引。
- 每个账户主体以 `platform_id + account_id` 为不可变主键；昵称只作为带时间的别名，
  重名不会自动合并。
- 每群的后台整理与反馈学习各有 24 小时 `500000` token 预算；预算不足的维护任务延期，
  不调用模型。在线 Reader 不占这两类后台额度，独立记录用量；Reader 失败时不注入记忆，
  AstrBot 主回答链继续正常运行。控制台可分别重置后台有效额度，原始账本不会删除。
- 日志默认不输出聊天正文。
- 不连接或重启消息适配器，不发送消息。

线上配置可以用 `allowed_umos` 限定群范围；留空明确表示所有群。UMO 的首段必须是
平台实例 ID，例如 `synthetic_bot:GroupMessage:123456`，不是 `aiocqhttp` 这样的适配器类型。

## 当前结构

```text
AstrBot event hook (main.py)
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
L0 RequestSnapshot (scope / cutoff / row bound / revision vectors)
        |
        v
host-bound identity/reply anchors + query-aware activity retrieval
        |
        v
snapshot-bounded alias + trigram FTS + embedding candidate retrieval
        |
bounded episode / semantic / graph expansion
        |
        v
deduplicated EvidenceAtomPack + source fingerprint audit
        |
        v
one strict evidence-reader.compact-host-speaker semantic pass
        |
        v
source re-audit + bounded memory surface for the main AstrBot LLM
        |
        +-- bounded background distillation and feedback maintenance
        +-- persistent bounded operational state and resumable maintenance jobs
```

`mr_memory/` 不依赖 AstrBot，可以通过普通 Python 测试和 JSONL 回放独立运行。
`main.py` 负责把 AstrBot 事件转换为核心模型。论文 Table 4 的七类工具和可塑关联、
重复媒体扩展工具只用于离线研究与后台整理；主 LLM 只接收有长度上限、可回到原始消息的
本地证据 envelope，不暴露查询时远程记忆工具。

插件调用自己配置的 provider，不继承当前会话主 LLM。后台增量整理与反馈学习由该独立模型
维护情节、人物事实、竞争释义和行为通路；普通回答在不可变 `RequestSnapshot` 下由本群
SQLite 与本地 embedding 生成候选包。查询人物时以 `platform_id + account_id` 为身份真值，
文本称呼在 Reader 内和其他证据一起推理、同名保持歧义；时间分析只使用 surface 中可审计的
Asia/Shanghai 样本。相似查询不会复用
旧答案，同一个 AstrBot event 的重复 hook 只复用一次本地读取结果。

embedding 只提供候选先验，不用相似度裁决语义。派生记忆、episode 摘要和图连接必须带原始
来源，Reader 先产出严格 source-backed 证书，主模型再决定如何回答。`NO_LOCAL_EVIDENCE`、
`IDENTITY_AMBIGUOUS` 与运行错误分别记录；本地读取错误不会被写成“没有记忆”。每次调用记录
本地阶段耗时、Reader prompt 长度、来源、截断和 provider usage；没有账单证据不估算 cost。

schema 17 新增 snapshot-indexed alias observations 与关键复合索引，并保留请求快照、
证据包缓存、语义证书、证书依赖和重建任务的持久化生命周期；
升级只迁移插件自己的按群 SQLite。`requirements.txt` 直接声明 NumPy，用于对 SQLite 中
完整候选向量做精确本地评分；Harrier 部署再额外使用 `requirements-harrier.txt`。可部署文件边界见
[运行时文件清单](docs/RUNTIME_FILES.md)。

消息按 per-message checkpoint 增量整理：批次只选择最早的 `PENDING/FAILED` 消息，附带
只读重叠上下文，成功后逐条推进；编辑或撤回会使派生图失效并重新排队。达到消息阈值或
维护周期后由有界后台 worker 池处理，不阻塞主回复。后台整理使用独立长超时。自动 worker
只处理 AstrBot 事件钩子实时观察到的 `LIVE` 消息；不调用任何消息适配器的历史或群列表 API。一次性
`BACKFILL` 只能由管理员显式调用，启动恢复和周期 sweeper 都不会自行消费它。DeepSeek V4
整理默认开启完整思考并流式消费响应；输出上限
跟随当前 DSV4 的 384k 能力，不再用 4k/8k/32k 截断冒充成本控制。管理员仍可执行 `/mrmem distill`
立即处理下一批。发布默认每批最多 80 条目标消息并附带 12 条只读重叠上下文，以兼容上下文
能力未知的 AstrBot Provider；长上下文部署可以显式增大，本项目线上回填经 320/500 条实测后
使用 500。进程若在批次中断，下一次打开数据库会把悬挂 checkpoint 标为 `FAILED`，
记录错误且不自动重试。
LLM 必须为每条目标消息提供图证据或放入紧凑的批内忽略 ID 列表，不能静默漏掉。
批内消息和账户主体只以 `mN`/`pN` 短 ID 暴露给模型，宿主验证后再无损还原真实证据键；
普通文本组件不再与 `plain_text` 重复发送。

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

插件安装后，AstrBot 的“插件页面”会自动出现“群聊记忆”。控制台与 AstrBot
仪表盘共用登录鉴权，无需开放额外端口，提供：

- 用最近 24 小时真实数据判断回答前回忆与反馈学习是否达到时延目标；
- 逐次显示任务、结果、耗时、Token 与“一次判断/判断后深挖/主动深挖/反馈学习”路径；点击任意
  一行首先查看由激活账本确认的持久记忆节点、记忆连接和读写状态，证据、处理步骤与原始结果
  只保留在默认折叠的审计账本中；
- 显示反馈积压、待验证行为记忆、额度等待和最久等待时间；
- 直接显示生效范围、回答前回忆、整理触发、反馈窗口与本地语义检索模型；
- 分别重置当前群的深挖/整理额度和反馈额度，同时保留不可变的原始账单；
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

`pilot` 子命令进一步固定同一候选包，比较 `cache`、0.16 单次语义 gate（`b16`）
和从第一步开放只读图工具的 `full-mr`。源数据库只读备份，迁移仅作用于实验副本；
每个 arm/repetition 独立克隆，并记录 attempted/completed/failed、payload/options hash、
Token、时延和 cutoff 审计。实验可恢复，但恢复时会拒绝候选、gold 或协议参数漂移。

首个真实调用实验的方法、消融矩阵和 token 结果见
[开发记录](docs/DEVELOPMENT_LOG.md)。论文机制当前覆盖范围与缺口见
[论文覆盖矩阵](docs/PAPER_COVERAGE.md)。实时、缓存与主动重建的当前证据、预注册指标和
成本边界见[运行时研究报告](docs/research/MR_MEMORY_RUNTIME_STUDY.md)。

## 反馈学习闭环

反馈维护按 proposal ID、来源修订号和内容指纹冻结每批任务。失败只将本次实际处理且仍为
同一修订的待分析条目标为 `FAILED`，不会覆盖已提交、已忽略或已修订的结果，也不会自动
重试失败批次。若准备阶段尚未触及任何条目便失败，该完整批次暂停，条目保留 `PENDING`；
待分析数量包含这些暂停项，不表示它们正在运行。后续不同批次仍可继续；旧失败作业及错误
保留。修复原因后可由管理员核对冻结来源并定向人工恢复，插件不自动重开历史失败任务。

启用后，插件为主 Agent 的每次请求保留不含隐藏思维链的可观察工作图。短时间连续反馈先
等待 15 秒合并，每次最多六条。独立模型在一次完整推理中同时判断后续消息是否确实评价了
某个有界 trace，并选择忽略或生成最小修改计划；不会为了第二次判断而串行调用模型，也不会
在提交前逐条运行 Harrier 关联检索。宿主继续校验 proposal、群、发送者、时间、trace、证据
键和修改范围。低于启用阈值但可归因的结果保留为 `PROVISIONAL`，不会进入回答；后续同向
证据累计足够才晋升为 `ACTIVE`。主 LLM 只接收临时、有限且已启用的行为提示。

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
`.dev/`，不会进入仓库。启用线上潜意识或显式运行真实 Provider 实验时，有界证据包会发送
给所选 LLM Provider；离线 `cache` arm 不调用 Provider。执行真实数据实验前应确认该
Provider 和数据外发范围符合部署者的隐私要求。

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
