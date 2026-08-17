# 从检索到重建：面向真实群聊 Agent 的实时、缓存与反馈驱动图记忆研究

> **状态：研究初稿，2026-08-17。** 本文把已经复核的线上账本、人工证据检索集和历史遮罩实验，与尚未完成的实验严格分开。表中 `TBD` 表示数据尚未产生；它不是零、失败或可由已有案例外推的数值。

## 摘要

长期群聊记忆的困难不只是从大量历史消息中“搜到相似文本”，还包括判断玩笑与事实、把变化中的昵称绑定到稳定账号、沿多跳关系重建群内语义，以及在后续反馈到来时修订曾经激活的错误通路。本文研究一个按群物理隔离的图记忆插件：宿主保存原始证据和确定性身份关系，本地检索器只初始化候选，独立于主聊天模型的潜意识 LLM 负责查询时语义判断和图遍历，可塑语义层则保存竞争释义、反馈效用与带来源的修订。

本文首先审计线上 0.16 回答前路径，并将其定义为弱基线 **B0**，而非合格方案。在一个经复核的单群观察窗口中，54 次调用有 49 次完成、5 次失败；25 次返回记忆简报、24 次返回无相关记忆。54 次全部走 `fast` 路径且只执行 `host_prefetch`，没有一次升级到图工具循环。该窗口共记录 1,258,333 Token；按“54 行、两条无 usage timeout 记测得零”的口径，单次均值 23,302、中位数 21,315.5、线性插值 p95 为 43,086.7。端到端墙钟均值 20.61 秒、中位数 7 秒、p95 为 80.35 秒，最长 90 秒。五次失败包括三次 JSON `ValueError` 和两次超时。由此可见，B0 虽然重新引入了独立 LLM，却在该窗口中实际退化为“预取后一次判定”，既未表现出主动重建，也存在明显长尾和协议失败；这些运行没有人工语义标签，因此不提供语义准确率。

候选层结果则更积极：在 47 条人工批准查询、24,804 条候选消息和 132 条正例证据标注上，Harrier 的 MRR@20、Hit@20 和平均 Evidence Recall@20 分别为 0.8413、1.0000 和 0.6652；MiniLM 分别为 0.6411、0.8511 和 0.5170；字符/二元组 BM25 分别为 0.8357、0.8936 和 0.5649。当前聚合保存的配对 bootstrap 显示，Harrier 相对 MiniLM 的 MRR 与 Evidence Recall 差值区间均高于零；Harrier 与 BM25 的 MRR 差值为 0.0056，95% 区间跨零。Hit@20 目前只有描述性差值，没有在可提交聚合中保存配对区间。该结果只证明候选覆盖，不证明语义终判。

既有高级语义和反馈案例不能承担总体结论。“好女孩”实验的旧评分没有要求模型识别其作为侮辱性讳称及被群友再次玩笑化的语义，因此高分主要反映主题聚合，构念效度不足；反馈实验只有两个手选行为样本，其中一个存在 control 天花板；真实调用 #726 的旧 r4 又把一条研究者重新标注为“支持带保留的口嫌体正直”的时序证据链判为完全无依据。该重标仍待独立盲审，只能作为诊断标注。本文因此预注册本地缓存 **A1**、完整主动重建 **A2** 与版本化混合路径 **A3** 的真实调用遮罩实验。新 pilot 尚未跑完，本文不预报胜负。当前最可靠的结论是：本地检索已经足以生成较强候选，但 B0 仍是未获质量验证的不及格弱基线，既有语义评测也不能证明其合格；实时重建相对缓存的收益、代价与适用边界仍需由盲评和配对实验决定。

**关键词：** 长期对话记忆；图记忆；主动重建；群聊语义；反馈修订；实体绑定；缓存；检索增强生成

## 1. 引言

### 1.1 群聊记忆不是静态文档检索

检索增强生成通常把问题映射到一个或多个文档，再由生成模型根据这些文档作答 [1]。这种范式适合稳定语料中的显式事实，却不足以描述长期群聊中的四类变化。

第一，群内表达高度依赖上下文。相同短语可能在不同阶段表示原义、反话、讳称、表情包索引或对既有梗的再度戏仿。第二，人物的稳定标识通常是平台账号，而群昵称、称呼和自称随时间变化；只按文本实体合并会把同名账号混为一人。第三，群友会纠正模型、撤回说法或用明显负面反馈否定此前回答，记忆系统必须保留原始证据并改变当前活动视图。第四，聊天是实时交互：即使一个更深的图遍历在离线评测中更准确，如果它在每次请求前引入一分钟级延迟，也可能不可用。

因此，群聊记忆需要同时解决候选召回、语义重建、主体绑定、记忆修订和运行成本。把其中任一层的成功替代为整个系统的成功都会产生错误结论。例如，embedding 命中正确消息不等于最终模型采用了它；图的最大连通分量很大不等于关系正确；LLM 输出了一条反馈决策也不等于可塑图真的发生了有效修改。

### 1.2 本文研究的问题

本文围绕六个研究问题展开：

- **RQ1：候选召回。** 本地 embedding 与词面检索能否在严格群作用域和时间 cutoff 下覆盖人工确认的原始证据？
- **RQ2：运行时重建。** B0 的一次语义判定、本地缓存 A1 与完整主动重建 A2，在证据召回、事实一致性、正确保留疑虑和无依据外推上有何差异？
- **RQ3：实时与缓存。** 哪些状态可以安全物化，哪些查询必须在调用时重新分析；不同选择的延迟、Token、陈旧错误和质量收益如何？
- **RQ4：高级图链接。** 竞争释义、动态关系类型、多跳路径和反馈效用能否提高群内梗与人物语义的正确选择，而不是只增加节点和边？
- **RQ5：反馈修订。** 后续人类反馈能否纠正实际参与过错误回答的路径，并避免修改无关记忆或确定性主体事实？
- **RQ6：主体绑定。** 宿主账号锚点与模型提出的语义别名如何低耦合协作，同时保持重名、改名和纯文本新昵称的不确定性？

### 1.3 贡献与边界

本文目前可以确认的贡献是：

1. 给出一个按群物理隔离、证据可追溯、身份层与可塑语义层分离的系统设计；
2. 建立真实调用严格 cutoff 的遮罩回放方法，并保存候选、工具步骤、source key、公开简报、Token 和时延，而不保存隐藏思维链；
3. 将候选召回、重建控制、最终回答、反馈修订与成本分别度量，避免以单一案例替代端到端结论；
4. 公开记录负结果：B0 在已审计窗口中从未升级到图遍历，“好女孩”旧 rubric 构念不足，#726 的研究者诊断标注指出旧 r4 的语义结论可能不成立，但该标注仍待独立盲审。

本文不声称已经完整复现 MRAgent [2]，也不声称当前插件已具备合格的群聊语义理解。原论文公开 benchmark 尚未复跑，新版 A1/A2/A3 真实调用 pilot 也尚未完成。

## 2. 相关工作

### 2.1 从 RAG 到图检索

RAG 将参数化语言模型与外部非参数记忆结合，改善知识密集型生成的可追溯性和可更新性 [1]。GraphRAG 进一步通过实体图和社区摘要处理面向整个语料的全局问题 [3]。HippoRAG 使用知识图和 Personalized PageRank 进行单步、多跳检索，强调图结构相对反复 LLM 检索的效率 [4]。这些方法说明“召回哪些文本”和“如何组织文本间关系”是两个不同问题，但它们并不自动解决长期对话中的主体变化、玩笑语义与反馈修订。

### 2.2 主动重建与 agentic memory

MRAgent 将记忆表示为 Cue--Tag--Content 图，并让 LLM 在访问过程中迭代探索和剪枝，而不是先静态检索再统一推理 [2]。本文沿用其核心区分：embedding 只提供候选先验，语义相关性和是否继续遍历由重建控制器决定。但原论文同时明确承认其构建是静态的，不做随时间更新、巩固或遗忘，图会随交互单调增长；因此本文的反馈可塑性不是“复现论文后自然得到”的能力，而是需要独立验证的扩展。A-MEM 通过类 Zettelkasten 的动态链接让新记忆触发现有记忆的属性与关系更新 [5]；MemGPT 则从分层存储和上下文分页角度处理长期记忆 [6]。

### 2.3 时间化、生产化与图记忆

Zep/Graphiti 把对话和结构化数据统一到时间感知知识图中，并在 LongMemEval 等任务上评估时序更新 [7]。Mem0 比较了基础记忆和图记忆的效果与运行成本 [8]。这些系统提供了重要工程参照，但其公开实验通常围绕双人助手或合成长对话；真实多人群聊中的账号主键、群作用域、讳称和群体梗仍需要单独评估。

### 2.4 长期记忆 benchmark 与知识更新

LoCoMo 覆盖长程问答、事件总结和多模态对话；其人工错误分析还把幽默/讽刺误解和错误说话人归因列为独立问题，直接对应本文的群聊语义与主体风险 [9]。LongMemEval 把能力分为信息提取、多会话推理、时间推理、知识更新和 abstention，并把系统设计拆成 indexing、retrieval 与 reading 三阶段 [10]。MQUAKE 则指出，一次知识修改即使能回答直接改写，也可能无法传播到依赖该事实的多跳问题 [11]。本文借鉴这些维度，但使用真实群聊调用和原始 source key 作为证据。由于语料不可公开，本文的内部结果不能与公开 benchmark 分数直接比较。

## 3. 系统设计

### 3.1 分层状态与信任边界

对群作用域 (s)、当前请求时间 (t) 和查询 (q_t)，系统维护：

- cutoff 前的原始证据集合 $E_{s,<t}$；
- 由宿主账号字段确定的主体集合 $P_s$；
- Episode、Semantic Claim、Topic、Cue 和 Tag 组成的蒸馏图 $G^d_s$；
- 保存竞争释义、行为规则和反馈效用的可塑图 $G^p_s$；
- 查询期间临时活动的工作状态 $Z_t$；
- 最终交给 AstrBot 主 LLM 的有界公开简报 $B_t$。

每个群使用独立 SQLite 文件，群作用域来自当前事件的 AstrBot UMO，不能由 LLM 参数或用户查询指定。所有图查询和 source 展开都二次绑定该作用域。历史遮罩实验还要求任何候选满足 `document.sent_at < query_time`，防止后续消息、后续 topic 修订或未来别名泄漏到过去调用。

### 3.2 真值层、主体层与语义层

原始消息、撤回、编辑、reply、mention 和 Bot 可见回复属于真值层。主体主键为：

\[
\text{Participant}=\text{scope}+\text{platform}+\text{account\_id}.
\]

昵称只作为带时间的 alias。结构化 mention/reply 可以把一条消息连接到目标账号；普通文本新称呼只有在当前群唯一命中既有别名时才能成为主体候选，否则保持 unresolved。LLM 可以在可塑语义层提出“某称呼通常指向某角色/概念”的关联，但不能据此合并两个账号，也不能跨群推断现实人物同一性。

这种分层刻意限制了 LLM 的写权限。确定性身份边和原始来源不能由模型修订；模型可修改的是有 source key、认知状态和版本记录的局部语义关系。

### 3.3 构建与候选初始化

后台构建把消息整理为 Episode、结构化 Claim、Topic 和关联边。Claim 区分 speaker 与 subject，并保留 `ACTIVE`、`UNRESOLVED`、`QUARANTINED`、`CONFLICTED`、`SUPERSEDED`、`RETRACTED` 和 `STALE` 等状态。单来源高风险身份或权限说法不会直接进入自动检索视图。

本地 embedding 为图单元和必要的原始证据建立向量，只负责生成候选排序。词面 BM25 仍作为便宜且可解释的基线。候选分数不是事实置信度，也不是“是否需要潜意识 LLM”的死阈值。

### 3.4 查询时控制器与实验 arms

本文固定以下术语，避免把版本号当作效果结论：

| Arm | 定义 | 语义决策位置 | 预期风险 |
| --- | --- | --- | --- |
| C0 | 仅主 LLM 最近上下文，无长期记忆 | 主 LLM | 长程证据缺失 |
| R0 | raw top-k 候选直接注入 | 主 LLM | 噪声、缺少多跳和状态 |
| **B0** | 0.16 weak baseline：宿主预取后由独立 LLM 返回 `brief / none / escalate`；只有 escalate 才开放图工具 | 一次预判，例外升级 | 过早停止；把主动重建变成罕见路径 |
| **A1** | versioned local cache：本地物化有来源的工作记忆；作为离线消融，不把规则当语义真值 | 缓存 + 主 LLM | 压缩损失、陈旧、主 LLM负担 |
| **A2** | full MR：独立潜意识 LLM 从第一步就可使用只读图工具，自行选择继续、回退、综合或 abstain | 完整查询时 Agent | 高延迟、工具循环和上下文膨胀 |
| **A3** | hybrid：缓存候选与稳定工作状态，独立 LLM仍做语义裁决，并在歧义、修订或证据不足时完整遍历 | 缓存 + Agent | 唤醒策略本身可能形成新盲点 |

B0 被恢复到线上只是为了避免 A1 未经实验就替代语义 gate；它不是对 0.16 语义能力的认可。A2 更接近 MRAgent 的主动重建思想，但“工具更多、思考更深”也不自动等于答案更好，#726 的过度浏览历史已经给出反例。

### 3.5 可塑语义图与反馈修改

可塑节点限定为 concept、behavior、symbol、topic、preference 和 procedure。关系类型可注册、复用或带版本修订，边可强化、抑制、休眠、废弃和可逆合并。每次修改必须引用当前群、当前维护包内已检查的 source key。

反馈链路为：

```text
request/action/response
        -> later human feedback
        -> evidence-bound hypothesis
        -> signed credit for actually activated paths
        -> versioned plastic-graph mutation
```

负面情绪是“需要复核”的强信号，不是命题为假的充分证据。负向修改只能作用于目标回答 trace 中实际激活的可塑边；维护阶段后来发现的替代路径不能替旧回答承担责任。事实置信度与路径 utility 分开保存。原始证据与历史 revision 不删除，遗忘只把低效用节点移出活动视图。

### 3.6 实时与缓存

本文把缓存分为三个风险等级：

1. **embedding/document cache**：缓存确定性文本表示，语义风险最低；
2. **versioned graph working-state cache**：缓存带 graph revision、identity revision 和 source keys 的活动状态，风险来自图更新后的失效；
3. **final brief cache**：直接复用旧语义结论，风险最高，因为同义改写、时间变化或新的纠正可能改变结论。

缓存 key 至少必须绑定 scope、图 revision、身份 revision、模型/提示协议和查询表示。cache hit 只能表示计算可复用，不能表示旧简报在当前语境中仍然正确。A1/A3 实验将分别测量 warm hit 和 stale hit，而不是只报告命中率。

## 4. 方法

### 4.1 数据、隐私与证据等级

本文使用三类数据：

1. 47 条经人类明确批准的群聊检索问题，底层语料 24,804 条消息，共 132 条正例证据标注；
2. 真实 `/chat` 调用的脱敏 ledger 与严格 cutoff 遮罩副本；
3. 单元测试、图查询性能记录和线上运行遥测。

真实正文、数字账号、群号和数据库快照只保存在 Git ignored 的 `.dev/`。可提交的报告只包含聚合统计、脱敏案例和 artifact 的方法描述。生产 ledger 不保存隐藏推理，只保存查询/结果哈希、工具、参数、source keys、公开简报、Token 与时延。

证据等级按以下方式使用：人工批准检索集可以支持候选层比较；严格遮罩且主模型、上下文和 cutoff 一致的配对实验可以支持该样本内的反事实比较；线上观察窗口只能描述运行状态；单元测试只能证明约束可执行。任何单案例都不能外推总体效果。

### 4.2 候选检索评测

每个问题只在同一群且早于 query time 的消息中排序。本文报告：

- **MRR@20**：第一条正例在前 20 名内的倒数排名均值；
- **Hit@20**：至少一条正例进入前 20 的问题比例；
- **Evidence Recall@20**：每题进入前 20 的正例数除以该题全部正例数，再对问题求平均；
- **All-Evidence@20**：全部正例均进入前 20 的问题比例。

模型差值采用以问题为配对单元的 percentile bootstrap，固定随机种子 `20260817`，重复 20,000 次，报告 2.5% 和 97.5% 分位数。当前区间没有多重比较校正；47 题规模也不足以稳定分析所有记忆类型。

### 4.3 B0 线上观察窗口

B0 结果来自一个经复核的单群 ledger 切片，共 54 次调用。抽取条件为 `runtime_reconstruction.started_at >= 2026-08-06 19:21:33Z`，聚合产物生成于 2026-08-17 03:03:02.686370Z。`completed`、`brief`、`none`、`fast`、工具步骤和错误类型直接读取实验/usage/reconstruction ledger。Token 分布以 run 为单位；时延取实验开始至终态的墙钟。p95 使用线性插值。

上述 UTC cutoff 是观察窗口的查询下界，不是第一条调用的精确时间；隐私安全聚合已记录源 SQLite 快照的 SHA-256，以便后续确认统计输入未漂移。两次 timeout 在 Provider 返回 usage 前结束，因而“54-run measured”分布把这两行的已记录 Token 视为零；另行报告的有 usage 分布只覆盖 52 行。由此总量只能解释为**已测得成本**，不是账单上界。由于没有随机对照和人工语义标签，该切片不能用于比较 B0 与 A1/A2 的效果，也不能把 `brief` 或 `none` 当作正确性标签。

### 4.4 真实调用遮罩协议

每个待评调用执行以下步骤：

1. 冻结真实调用时间为 cutoff，移除所有更晚消息和派生状态；
2. 为该群构建独立 SQLite，所有 arm 共享同一份 cutoff 前图；
3. 由人工在查看模型输出前标记必需证据、允许结论、必须保留的疑虑、错误外推和正确主体；
4. 各 arm 使用相同主 LLM、系统提示、最近上下文和随机化顺序；唯一变量是记忆路径；
5. 保存每次候选、工具访问、source keys、公开 brief、回答、Token 和分段时延；
6. 再次验证所有访问 source 的时间严格小于 cutoff；
7. 对输出做盲评，自动 judge 只作辅助，不替代人工证据裁决。

pilot 的 provenance manifest 同时绑定 exact masked messages、固定 candidates 与隔离 base SQLite 的 SHA-256，并记录研究者对“该 base 仅由 cutoff 前派生状态构成”的显式 attestation。三类输入任一变化都会使既有运行不可恢复，从而防止样本、候选或图快照在 arm 之间静默漂移；但这种 attestation 不是对构建历史的独立密码学证明，也不能单凭 base hash 证明图中从未混入 cutoff 后派生状态。更强的确认性方案应在实验流程内部从 exact masked messages 重新构建每个隔离图，同时保存构建协议、输入 hash、派生日志与产物 hash，使图快照可由冻结消息重现，而不是依赖研究者对既有 base 的声明。

正式确认集的规模将在 pilot 后按方差和目标最小效应做 power 分析；当前操作目标是不少于 60 个真实调用，并覆盖直接事实、时间更新、多跳、群内梗、主体歧义和 no-memory 负例。开发过程中反复见过的“好女孩”“阿拉蕾”和 #726 只能作为诊断集，不能进入无污染 holdout。

### 4.5 端到端指标

重建层报告 relevant/none/escalate 混淆矩阵、false-none、missed-escalation、工具步数、重复访问率、gold source 覆盖和最终证据路径长度。最终回答层报告：

- source-supported claim precision；
- required evidence recall；
- unsupported named fact rate；
- correct abstention；
- identity/subject binding accuracy；
- 人工帮助度和疑虑保留评分。

反馈层同时测量 efficacy、paraphrase/generalization、specificity、reversal 和 collateral activation。系统层报告 cold/warm p50、p90、p95、首块与总时延、普通输入/cached input/输出 Token、cache hit/stale hit、失败率和每个“盲评确有增益”回答的成本。

当前 pilot 的 Provider arm 通过 OpenAI-compatible 接口直连模型，采用非流式请求并设置 `max_retries=0`；其时延从请求发出计到完整响应或错误返回，只表示受控 API 调用的总耗时。它既不是 AstrBot 在线路径包含事件调度、插件钩子、流式传输与宿主重试后的端到端时延，也不是首 Token/首块时延。因而 pilot 可用于同一协议下比较 arm 的模型调用成本和总等待时间，不能直接替代线上 B0 遥测或据此承诺用户可见响应延迟；B0/0.16 仍只是不及格且未经语义质量验证的弱基线。

二元配对结果计划使用 McNemar 检验；连续或等级差值使用配对/cluster bootstrap。相同真实调用的多次采样不是独立样本，统计单位仍是原调用；跨群汇总时按群或调用 cluster。

## 5. 结果

### 5.1 本地候选已经显著强于 MiniLM，但没有取代词面基线

表 1 给出 47 个问题上的检索结果。Harrier 对 MiniLM 的优势主要体现在第一条证据和前 20 的证据覆盖；BM25 的 MRR 与 Harrier 接近，但遗漏更多“至少一条证据”和完整证据集合。这说明语义向量与词面检索具有互补性，也说明不能仅凭 MRR 选择运行时方案。

**表 1　人工批准群聊检索集，47 queries / 24,804 documents / 132 positive evidence annotations**

| 检索器 | MRR@20 | Hit@20 | Mean Evidence Recall@20 | All-Evidence@20 |
| --- | ---: | ---: | ---: | ---: |
| MiniLM-L12-v2 | 0.6411 | 0.8511 | 0.5170 | 0.2553 |
| BM25 | 0.8357 | 0.8936 | 0.5649 | 0.2340 |
| Harrier-270M | **0.8413** | **1.0000** | **0.6652** | **0.3617** |
| Harrier-270M + `group-memory` prefix | 0.7632 | 0.9787 | 0.6617 | 0.3830 |

**表 2　Harrier 的配对差值与 95% percentile bootstrap 区间（20,000 次，seed 20260817）**

| 比较 | 指标 | 差值 | 95% 区间 |
| --- | --- | ---: | ---: |
| Harrier − MiniLM | MRR@20 | +0.2002 | [+0.0752, +0.3271] |
| Harrier − MiniLM | Evidence Recall@20 | +0.1482 | [+0.0596, +0.2376] |
| Harrier − BM25 | MRR@20 | +0.0056 | [−0.0990, +0.1120] |
| Harrier − BM25 | Evidence Recall@20 | +0.1004 | [+0.0142, +0.1933] |
| Harrier `group-memory` prefix − default | MRR@20 | −0.0782 | [−0.1398, −0.0271] |
| Harrier `group-memory` prefix − default | Evidence Recall@20 | −0.0035 | [−0.0638, +0.0461] |

Harrier 相对 BM25 的 MRR 区间跨零，不能写成“整体排序显著更好”。其较可靠的现有优势是 k=20 时描述性的 Hit@20 更高，且 Evidence Recall 的配对区间高于零。另一方面，给 Harrier 人工添加 `group-memory` 任务前缀使 MRR 下降 0.0782，区间完全低于零，而 Evidence Recall 差值接近零且区间跨零；任务提示不是无风险的“领域增强”，必须当作独立检索器版本评估。

该测试集由回复链候选起步，可能偏向词面可追溯问题；词面 decoy 也未被人工验证为负例，因此无法从这些表计算全体候选上的 precision。只有 47 题，区间也未按记忆类型分层。最合理的工程含义是保留 BM25/词面信号并考虑融合，而不是把 Harrier 分数或人工任务前缀升级为语义真值。

### 5.2 B0 在 54 次真实调用中从未发生主动图重建

**表 3　B0 单群观察窗口的运行结果**

| 指标 | 结果 |
| --- | ---: |
| 调用数 | 54 |
| 完成 / 失败 | 49 / 5 |
| 完成率 / 失败率 | 90.74% / 9.26% |
| brief / none / failed | 25 / 24 / 5 |
| 完成调用中的 brief / none | 51.02% / 48.98% |
| `fast` 路径 | 54 / 54 |
| 仅 `host_prefetch` | 54 / 54 |
| escalation / 图工具循环 | 0 / 54 |
| 错误 | 3 × JSON `ValueError`; 2 × timeout |

B0 的设计允许 `escalate`，但该窗口的经验行为是 0 次升级。换言之，独立 LLM 的存在并没有使这 54 次调用成为主动重建；它只是对宿主预取包做一次较昂贵的 brief/none 判定。这个结果不能证明所有请求都不需要图遍历，因为当前没有每次调用的人工“是否应升级”标签。相反，0/54 是必须优先检验的失败假设：prompt 中“escalate 是例外”的叙述，可能让模型在群内梗、多跳关系和主体消歧上过早收敛。

五次失败也不能统一归因于模型“思考太久”。三次是输出协议/JSON 解析错误，两次才是超时。正确优化需要分别测量语义控制、协议修复与 Provider 时延，不能只缩短输出或关闭思考。

### 5.3 B0 的时延尾部很长，Token 成本尚无质量收益分母

**表 4　B0 每次调用的 Token 与端到端墙钟**

| 分布 | Mean | Median / p50 | p95（线性插值） | Max |
| --- | ---: | ---: | ---: | ---: |
| Measured ledger Token / run（n=54；2 条缺失记 0） | 23,302 | 21,315.5 | 43,086.7 | 46,854 |
| Usage-present ledger Token / run（n=52） | 24,198.7 | 21,450.5 | 43,220.9 | 46,854 |
| Wall latency / run | 20.61 s | 7.00 s | 80.35 s | 90.00 s |

54 次合计记录 60 个 Provider 调用、1,258,333 ledger Token，其中 ordinary input 1,013,599、cached input 136,576、Provider-reported output 108,158。时延均值约为中位数的三倍，说明分布具有明显慢尾；90 秒最大值还受到线上超时边界影响。表中 Token/run 使用 54 行并把两条没有 usage 的 timeout 视为“测得零”，因此 23,302 不是这 54 次调用的真实账单均值。只在 52 条有 usage 行上计算时，均值为 24,198.7、中位数 21,450.5、p95 为 43,220.9。

聚合只能区分 ordinary input、cached input 和 Provider-reported output，不能再拆 reasoning 与可见输出；三条 run 标为 `repair_attempted`，但相关调用也没有从重建 phase 中单独剥离。因此不能从总量推导具体货币账单，也不能精确计算五次失败浪费了多少 Token。

更关键的是，B0 用这些成本换来的仍然是 54/54 仅预取、0/54 工具升级。只有在人工盲评证明“一次预取判定已足够”的请求上，这笔成本才可能合理；目前尚无该证明。

### 5.4 #726 的研究者重标推翻了旧 r4 的成功叙述，但仍待盲审

真实调用 #726 询问某位群成员是否对一类游戏“口嫌体正直”。重新复核后的 researcher-labeled 诊断标注不是一句孤立旧发言，而是一条时间链：同一账号先前表示自己把类魂游戏“玩吐”，后来又询问相关游戏与购买问题，讨论其类魂、PvE/PvP 特征，评价“好玩”，并提到“刚开服”。这组证据被研究者标为支持**带保留地**概括为口嫌体正直：它表明先前厌倦与后续重新投入并存，但不能证明明确动机，也不能把“抢首发”写成当事人原话。该标注的 provenance 明确为 `researcher_labeled_pending_blind_human_review`，所以它是开发诊断，不是最终人工 gold。

旧 r4 一方面正确指出没有“抢首发”的直接证据，另一方面却把后续购买与投入也概括为 entirely unsupported。按该诊断标注，后者是语义失败。旧实验中 r4 通过宿主直接证据门把重建 Token 从 r3 的 37,316 降到 5,480，下降 85.31%；但成本下降伴随关键结论被错误否定，不能继续作为“质量不变的效率优化”证据。即使未来盲审改变最终语义等级，r4 漏读同一账号后续购买/游玩证据这一事实仍成立。

这个反例揭示了两个问题。第一，gold 必须覆盖时序行为链，而不能只标一句支持或反对文本。第二，宿主门只能决定“证据足以进入综合”，不能以词项重叠或单个候选替代 LLM 对整条命题的语义判断。#726 将保留为开发诊断案例，不进入未来无污染 holdout。

### 5.5 “好女孩”旧高分衡量的是主题召回，不是核心语义理解

既有可塑语义实验把“好女孩”重建为 MyGO 表情包梗、对 Bot 是否听话的戏谑标签和随语境变化的鉴定/调侃语，旧 rubric 因而给出 4/5 或 5/5。用户复核指出，测试真正困难的目标是识别“好女孩”在该群语境中可作为“臭婊子”的讳称，并进一步理解群友在知道这层指向后进行的再次玩笑化。

旧 rubric 没有把这一释义列为必要命中项，也没有要求模型在竞争释义之间根据当前上下文选择；评分还是同一潜意识模型基于该 rubric 的单次 judge，而非独立人工盲评。因此原 4/5、5/5 高分存在重大构念效度问题。它最多证明系统聚合了与该短语相关的多个主题，不能证明系统理解了目标语义。旧成功产物记录 37 次模型调用、221,659 Token；同一实验线另有可计量的 4,397 与 95,717 Token 失败尝试，以及两次未持久计费的中止调用，所以当时已知开发成本至少是 321,773 Token 加两次未知调用，而不是 221,659 Token。

后续“深层语义”实验进一步验证了这个负结果。它把视觉盲读、直接回答、竞争假设和“异常词形展开→时间竞争→图重建→反方证伪”分开，共 14 次调用、44,957 Token、累计 API 时间 198.0 秒；其中最深的 D 链为 5 次调用、14,220 Token、28.6 秒。D 链以 0.95 模型置信度把异常词形“臭表字”还原为“臭婊子”，并发现再引义候选，但仍认为 23 秒邻接不足以确认“好女孩→臭婊子”的同义边。这个结果应表述为**候选释义发现成功、核心关联未确认**，不是“好女孩测试通过”。

新的测试必须把下列状态同时表示在图中：

- 表层褒义；
- 特定作品/表情包来源；
- 侮辱性讳称假说；
- 对该讳称的二次戏仿；
- 每条释义的来源、时间、说话人群体、认知状态和竞争关系。

评分对象应是“当前查询激活了哪条释义及其保留意见”，而不是“回答中提到了多少相关梗”。由于该案例已被反复用于开发，它只能用于机制回归；新结论必须来自未见过的群内表达 holdout。

### 5.6 反馈与可塑图只有局部行为证据，尚无一般修订结论

现有反馈 A/B 包含两个手选行为样本。`no_followup` 中，当前 control 已经 3/3 避免旧问题，memory 也是 3/3，因此没有可测增益；`forced_choice` 中，control 为 0/3，memory 为 3/3，说明一条 sender-scoped 语义行为规则在该例中可以跨词面改写生效。两个样本的一次性维护与激活共 12,310 Token，最终回答 arms 共 16,995 Token。

这些结果有四个重要限制：

1. 两个样本均为手选案例，每个 arm 的三次采样共享同一底层调用，不能当作六个独立用户样本；
2. 一个样本存在 control 天花板，真正观察到改善的只有 `forced_choice` 一例；
3. 评分只检查指定行为，没有测量回答事实性、过度服从或跨任务副作用；
4. 旧线上维护允许 `graph_mutations=[]`，对 166 条历史决策的审计中没有一次显式前向图修改。后续由宿主根据已提交 hypothesis 物化边的兜底能改善工程闭环，却会混淆“LLM 主动学会修改图”与“宿主替它补写图”这两个主张。

因此，目前只能说反馈协议、scope 门禁和单例行为激活能够运行。不能说反馈系统已经实现通用知识修订、类 RLHF 学习，或能够可靠纠正群内事实。下一轮实验必须同时报告修改有效性、同义泛化、无关记忆保持、错误反馈抵抗和可逆性。

### 5.7 A1/A2/A3 新 pilot 尚未完成

以下表格是预注册结果位，不包含预测值。当前第一阶段 harness 只覆盖同一固定候选集上的 A1/cache、B0/B16 与 A2/full-MR；C0、R0、A3、主 LLM 最终回答盲评和多调用确认集尚未实现或运行。单案例 smoke 即使成功也只能验证 harness，不会填入总体效果格。

已经完成的唯一新版 smoke 是一条 researcher-labeled 诊断调用上的 cache-only 运行：1/1 完成、0 次 Provider 调用、0 Token，所有访问来源均通过逐来源 cutoff 审计。它引用了两个必要证据组，`required_group_recall=1.0`，但 `gold_key_precision_lower_bound=0.1579`，表明确定性物化器能覆盖关键证据，同时向下游暴露了大量非必要来源。该数值只验证经研究者声明的 masked fixture 上的执行、来源审计与噪声诊断；它不能独立证明图的构建历史，也不能作为 A1 语义质量或跨调用效果。隐私安全聚合见 [`pilot_smoke_aggregate.json`](data/pilot_smoke_aggregate.json)。B0/B16 和 A2/full-MR 的真实 Provider 运行仍未发生。

**表 5　真实调用遮罩 pilot：缓存与主动重建**

| Arm | n | Evidence Recall | Supported-Claim Precision | False None | Unsupported Facts | Human Helpfulness | Token / run | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0：最近上下文 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| R0：raw top-k | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B0：0.16 weak baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A1：versioned local cache | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A2：full MR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A3：hybrid | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**表 6　高级链接、主体与反馈修订 pilot**

| 实验 | 样本数 | 主要指标 | 结果 |
| --- | ---: | --- | --- |
| 竞争释义选择 | TBD | context-conditioned sense accuracy；正确保留疑虑 | TBD |
| 多跳图路径 | TBD | gold path reachability；path precision；无效浏览 | TBD |
| 昵称与主体绑定 | TBD | merge precision；正确 abstention；subject accuracy | TBD |
| 反馈后修订 | TBD | efficacy；generalization；specificity；reversal | TBD |
| cache 陈旧性 | TBD | warm hit；stale hit；错误复用；invalidation latency | TBD |

在这些单元填满之前，本文不会声称 A1、A2 或 A3 优于 B0。尤其不能因为 A2“更接近论文”就预设其更准确，也不能因为 A1 更快就预设其语义损失可接受。

## 6. 成本—收益分析

### 6.1 成本必须按生命周期分账

系统成本至少分为：一次性历史导入、历史图构建、在线增量构建、每请求重建、反馈维护、本地 embedding 和主 LLM 因记忆简报增加的输入。一次性回填不应计入日常每请求成本，在线预算也不应被一次性迁移挤占。

本文把在线收益定义为盲评确认的增量质量，而不是“返回了非空 brief”。一个基础指标是：

\[
C_{\text{helpful}}=
\frac{\text{在线重建 Token}+\text{主 LLM 新增 Token}}
{\text{相对 C0 取得盲评质量增益的请求数}}.
\]

同时报告 $\Delta Q/\Delta\text{Token}$ 和 $\Delta Q/\Delta\text{Latency}$。如果一个 arm 召回更多证据却产生更多无依据断言，其收益不能用 recall 单独表示。

### 6.2 现有 B0 账本说明“每请求一次 LLM”并不便宜

B0 观察窗口按 54-run measured 口径平均每次记录 23,302 Token、20.61 秒，且 p95 达 43,087 Token 和 80.35 秒；只看 52 条有 usage 行时，Token 均值为 24,198.7。由于 54 次都没有进入工具循环，这不是“完整主动重建”的上界或下界，而是一次预取语义判定的已测成本。它提示两条相反但都需要实验的可能性：

- A1 若能保持质量，可能消除重复把稳定记忆交给 Provider 重述的成本；
- A2 若能显著改善 false-none、多跳和群内语义，较高成本可能仍然值得。

没有 A1/A2 配对质量结果时，不能只按 Token 选择其中之一。

### 6.3 缓存收益必须扣除陈旧错误

缓存实验将分别报告冷启动、图未变化的 warm query、同义改写、图 revision 后查询、身份别名变化和事实更正。A1 的主要风险不是 cache miss，而是 cache hit 后静默复用错误简报。因而其有效收益应写为：

\[
S_{\text{cache,net}}=
\text{saved latency/token}
-\lambda\,\text{stale-error cost}.
\]

系数 $\lambda$ 不在本文中拍脑袋指定；报告将同时展示原始质量和成本的 Pareto frontier，由使用者决定不同群和任务的容忍度。

### 6.4 当前不能给出可靠货币成本，也不能直接套用论文延迟

现有聚合账本已经拆分 ordinary input、cached input 与 Provider-reported output，但没有把 reasoning 与可见输出分开；两条 timeout 又在 usage 返回前结束，不同 AstrBot Provider 还可能有不同计费协议。本文因此先报告已测 Token 与时延。货币成本只能在正式稿中引用实验当日 Provider 的第一手价格，并明确模型、缓存折扣、币种和时间，不能用今天的价格回填历史调用。

公开论文数值也只能作为量级参照。MRAgent 在 LongMemEval 上报告每样本 118k Token、586.11 秒，但表注明确包含记忆构建与检索，不能解释为一次在线 `/chat` 的串行延迟 [2]。Zep 在 LongMemEval-S 上报告 gpt-4o-mini 的 1.6k 平均上下文与 3.20 秒延迟，而 full-context 为 115k 与 31.3 秒 [7]；Mem0 在 LoCoMo 协议下报告基础版总响应 p95 约 1.44 秒、图版约 2.59 秒 [8]。这些研究的模型、数据集、计时边界、硬件和构建摊销均不同，不能拿来证明本插件应达到某个延迟，也不能与 B0 的 20.61 秒直接做收益率除法。

## 7. 有效性威胁与稳健性

### 7.1 内部有效性

真实调用可能受到模型版本、Provider 排队、网络、缓存预热和同时运行的后台任务影响。在线 B0 窗口没有随机对照，只能作描述性审计。超时调用可能在返回 usage 前被截断，形成右删失成本。未来实验将使用隔离数据库、相同 cutoff、交替 arm 顺序和分段 ledger，避免把部署时段差异当作算法差异。

### 7.2 构念有效性

“好女孩”旧评分已经证明，一个看似完整的 rubric 仍可能没测到用户真正关心的语义。最终评测必须在看 arm 输出前冻结必要证据、竞争释义、允许保留和禁止外推。帮助度不能替代事实一致性，主题相关也不能替代正确语用解释。LLM-as-a-Judge 只能辅助筛查，不能为群内私有语义建立唯一真值。

### 7.3 外部有效性

47 条检索题和现有案例来自少数真实群，语言、成员关系和梗密度可能不代表其他社区。测试集由可回指 reply 链生成候选，也可能偏向容易追踪的证据。结果不能外推到私聊、企业知识库、跨平台身份或视觉表情语义。

### 7.4 统计结论有效性

47 题的 bootstrap 区间较宽，且没有多重比较校正。重复采样同一真实调用不能扩大独立样本数。未来报告以调用或群为 cluster，并同时给出效应量与区间，不把 p 值当作实际价值。高级语义类别的正式样本数由 pilot 方差与 power 分析确定。

### 7.5 数据与时间泄漏

后续 topic、claim 修订、别名和反馈可能覆盖旧状态。只限制原始消息时间还不够；构建、candidate search、每个工具和 cache revision 都必须遵守 cutoff。任何无法证明 graph revision 早于调用的样本都应排除，而不是作为成功案例保留。

当前 pilot 用 masked messages、candidates、base SQLite 的 hash 与 researcher attestation 检测输入漂移，但这一链条仍依赖研究者对 base 构造过程的声明。若未能从冻结消息在实验流程内重建并核对派生图，就不能把“hash 一致”表述成“cutoff 来源已被独立证明”；确认性实验应升级为可重放构建后再纳入统计结论。

### 7.6 开发集污染

“好女孩”“阿拉蕾”和 #726 已被用于多轮 prompt 与机制调试，不能作为确认性统计样本。它们保留用于回归和失败分析。最终 holdout 必须在冻结实现后才揭示，并记录任何因调试而被提前查看的样本。

## 8. 讨论

### 8.1 B0 的问题不是“模型不够深”，而是控制结构尚未被验证

0/54 escalation 说明 B0 在当前 prompt 和流量下没有发挥主动图访问能力，但不能仅凭该数字判断每次都应该遍历。真正的问题是缺少监督信号：哪些请求在 host-prefetch 中已经足够，哪些请求需要多跳、主体消歧或竞争释义。下一步应先建立这组标签，再比较 B0 的升级决策，而不是继续用超时、输出上限或固定相似度阈值调参。

### 8.2 可缓存的是证据状态，不一定是最终语义

embedding、规范化 source 展开、确定性主体和版本化局部图天然适合缓存。最终 brief 则依赖当前问题、时间和群内语用，复用风险更高。A3 的合理方向不是完全取消查询时 LLM，而是让缓存承担重复机械工作，让潜意识 LLM 专注于相关性、歧义、多跳和修订。是否能以一次短 semantic tick 达成这一点，仍需 A1/A2/A3 实验回答。

### 8.3 高级链接的价值在路径选择，不在图规模

节点数、边数、最大连通分量、平均度、聚类系数和 k-core 可以发现孤岛、枢纽与关系爆炸，却不能证明某条边有正确语义；本文也没有把任一时点的生产图快照冻结成可复核的定量结果。高级链接应以三个问题验收：gold evidence 是否更容易在有限步数内到达；当前查询是否激活正确的竞争释义；后续反馈是否能只修改实际导致错误的通路。复杂网络指标只作为结构诊断，并必须绑定 graph revision 与采样时点。

### 8.4 主体绑定必须比语义图更保守

群内可能用新昵称指向旧账号，但模型“意识到是同一人”和系统“确定为同一账号”不是同一命题。宿主 mention/reply/account_id 提供确定性锚点；纯文本称呼只能形成待确认语义假说。这样会牺牲部分自动 recall，却能避免两个同名成员的记忆不可逆混合。未来可以让潜意识层提出 alias proposal 并请求管理员确认，但不应允许可塑边直接覆写 Participant 主键。

### 8.5 反馈修改是必要能力，但强反馈仍不是直接真值

如果一开始把玩笑当事实，系统必须能够 supersede、retract 或降低错误通路效用。与此同时，人类负面回复可能是玩笑、情绪或针对表达方式，而不是事实纠正。稳健反馈系统需要保留 Action--Feedback--Hypothesis 链，区分事实 confidence 与行为 utility，并对高影响修改提供人工确认。它更接近带外部证据的在线记忆编辑，而不是对模型参数做 RLHF。

### 8.6 不保存图片和隐藏思维链是有意边界

图片持久化和多模态分析会显著增加存储、隐私与推理成本。当前系统只允许有界 opaque hash 统计重复媒体，不能从 hash 猜图意。隐藏思维链同样不持久化；可复核对象是候选、工具、source keys、公开 brief 和状态修改。这会限制事后解释模型每一步心理过程，但保留了足以做反事实审计的外部行为轨迹。

## 9. 结论

本文没有得到“实时 Agent 必然优于缓存”或“当前潜意识已理解群聊语义”的结论。已经复核的结果支持更窄、也更重要的判断。

第一，在这一个单群、47 题的候选检索集上，Harrier 相对 MiniLM 有明确优势，并在 k=20 时比 BM25 找到更多问题和更多证据；但 BM25 的 MRR 与 Harrier 无可靠差异，词面基线仍应保留。第二，没有证据把 B0 认定为合格语义方案：54 次线上调用全部止于 host prefetch、没有一次图升级，按“缺失 usage 计测得零”的 54-run 口径仍平均记录约 2.33 万 Token，并有 9.26% 协议/超时失败和 80 秒级 p95；但这些运行没有人工语义标签，不能据此计算准确率。第三，既有高级语义和反馈证据存在严重边界：“好女孩”旧 rubric 没测核心语义，后续深挖也只发现候选而未确认核心边；#726 旧 r4 按待盲审诊断标注失败；反馈改善只发生在一个手选行为案例。

因此，当前工程决策应是：线上先保留 B0 作为可观测弱基线，不把 A1 未经验证地设为语义终判；同时用严格遮罩、盲评和分段账本完成 A1 本地缓存、A2 完整主动重建与 A3 混合路径的配对实验。最终选择应依据 supported-claim precision、false-none、正确疑虑保留、p95 时延和每个真实质量增益的 Token，而不是依据架构直觉、图规模或单个漂亮案例。

## 参考文献

1. Lewis, P. et al. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS, 2020.
2. Ji, S., Li, Y., Hooi, B. [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036). arXiv, 2026. [官方实现](https://github.com/Ji-shuo/MRAgent).
3. Edge, D. et al. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). arXiv, 2024.
4. Gutiérrez, B. J. et al. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831). arXiv, 2024.
5. Xu, W. et al. [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110). arXiv, 2025.
6. Packer, C. et al. [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560). arXiv, 2023.
7. Rasmussen, P. et al. [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956). arXiv, 2025.
8. Chhikara, P. et al. [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413). arXiv, 2025.
9. Maharana, A. et al. [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753). arXiv, 2024.
10. Wu, D. et al. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813). arXiv, 2024.
11. Zhong, Z. et al. [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions](https://arxiv.org/abs/2305.14795). arXiv, 2023.

## 附录 A：指标定义与统计口径

| 指标 | 定义 | 主要误读风险 |
| --- | --- | --- |
| MRR@20 | 第一条 gold evidence 在前 20 的倒数排名均值 | 不反映其余证据是否找全 |
| Hit@20 | 至少一条 gold evidence 进入前 20 的查询比例 | 一条命中可能不足以回答多跳问题 |
| Evidence Recall@20 | 每题前 20 命中数 / gold evidence 数，再求均值 | 不衡量错误候选数量 |
| Supported-Claim Precision | 回答中可由本轮访问 source 支持的可验证 claim / 全部可验证 claim | 依赖 claim 分割和人工蕴含判断 |
| False None | 实际存在必要记忆但控制器返回 none 的比例 | 需要独立人工 gold |
| Missed Escalation | 预取包不足且图中存在必要证据，但控制器没有升级 | 不能由工具步数为零直接推出 |
| Stale Hit | cache 命中但旧状态已被 revision、身份变化或新证据实质失效 | cache hit 本身不暴露错误 |
| Correction Specificity | 修改后无关查询保持原正确行为的比例 | 只测目标问题会高估修订质量 |

B0 的完成率为 $49/54=90.74\%$，失败率为 $5/54=9.26\%$。brief/none 比例的 51.02%/48.98% 以 49 条完成调用为分母。54-run Token 均值把两条无 usage 的 timeout 记为已测得零，因此虽能与 ledger 总量 1,258,333 算术对齐，却不是实际账单均值；52 条有 usage 行的分布单独报告。聚合已拆 ordinary input、cached input 与 Provider-reported output，但仍需在新实验中按 gate、repair、tool loop 和 surface-answer phase 分账。

## 附录 B：预注册 arm 控制变量

所有端到端 arm 固定：

- 同一真实 query 与 cutoff；
- 同一群、同一隔离图快照；
- 同一主 LLM、system prompt、最近上下文和输出协议；
- 同一 construction 结果，不重复把一次性构建成本计入每个 arm；
- 同一人工 gold 和盲评 rubric；
- 交替/随机执行顺序；
- 相同失败重试政策，并把每次修复调用单独计费。

arm 之间只改变候选呈现、缓存状态和重建控制器。A2 可以自行停止，但从第一轮即可访问完整只读工具；B0 继续保留其一次预判/例外升级结构，以便测量控制器差异。A3 的唤醒规则在 pilot 前冻结，不能根据测试集逐题增加特例。

## 附录 C：主体与高级链接验收矩阵

| 场景 | 正确宿主行为 | 语义层允许行为 | 禁止行为 |
| --- | --- | --- | --- |
| 同账号改昵称 | 保持同一 Participant；记录别名历史 | 更新昵称使用语境 | 新建第二人物 |
| 两账号同昵称 | 保持两个 Participant；查询提示歧义 | 建立两个独立称呼候选 | 合并记忆 |
| mention/reply 新称呼 | 绑定结构化目标账号 | 增加有来源 alias hypothesis | 用显示名覆盖主键 |
| 纯文本新昵称 | 唯一别名命中才作候选，否则 unresolved | 提交待确认语义别名 | 无证据自动绑定 |
| 甲说乙的偏好 | speaker=甲，subject=乙 | 保存转述/不确定状态 | 写成甲的偏好 |
| 跨群同账号 | 各群独立 Participant | 不推断现实人物合并 | 跨库读取或连边 |
| 玩笑身份说法 | 保留证据并隔离高风险 claim | 形成竞争假说 | 当作确定性身份 |

## 附录 D：已有实验材料与复现入口

下列路径只描述仓库内实现与本地审计入口；真实正文和数据库保持 Git ignored：

- [论文机制覆盖矩阵](../PAPER_COVERAGE.md)
- [系统架构](../ARCHITECTURE.md)
- [反馈闭环](../FEEDBACK_LOOP.md)
- [主体模型](../IDENTITY_MODEL.md)
- [检索测试集协议](../RETRIEVAL_BENCHMARK.md)
- [开发与实验记录](../DEVELOPMENT_LOG.md)
- [隐私安全聚合结果](data/observed_evidence.json)
- [聚合分析脚本](../../scripts/analyze_runtime_study.py)
- [真实调用遮罩脚本](../../scripts/masked_ab_experiment.py)
- [Embedding 评测脚本](../../scripts/evaluate_embedding_models.py)
- [反馈 A/B 脚本](../../scripts/feedback_ab_experiment.py)
- [可塑语义实验脚本](../../scripts/plastic_semantics_ab_experiment.py)

正式稿需为每个结果保存：数据集指纹、代码 commit、模型标识、提示协议版本、运行时间、随机种子、arm 配置、逐调用 usage 和所有排除项。任何缺少这些字段的历史数字只能作为观察性背景。

## 附录 E：当前可声称与不可声称的结论

### 可声称

- 47 条人工批准查询上，Harrier 的前 20 证据覆盖高于 MiniLM；相对 BM25 的优势是描述性 Hit@20 更高且多证据 recall 的配对区间高于零，而非已证实的 MRR 优势。
- B0 单群 54 次观察窗口中没有发生一次 escalation，且存在明显时延长尾和 9.26% 失败；Token 只能按“已测得成本”解释。
- 当前系统已实现按群物理隔离、账号主体锚点、source-bound 图关系、查询 trace 和反馈 proposal 门禁。
- #726 的待盲审诊断标注、“好女孩”实验和既有反馈实验暴露了评测构念、控制器和归因边界。

### 不可声称

- B0 的语义能力合格；
- A1 缓存不会损失语义；
- A2 完整重建一定优于 B0；
- 已完整复现 MRAgent 或达到其公开 benchmark；
- 图越连通、节点越多，记忆质量越高；
- “好女孩”测试已经通过；
- embedding 高分等于证据真实或最终相关；
- 负面情绪本身证明旧记忆错误；
- 现有反馈等同于 RLHF 或通用在线学习；
- 主体系统已经解决所有纯文本新昵称；
- 单个遮罩案例能够证明总体质量提升；
- Token 下降必然保持回答质量；
- 当前系统具有可验证的人类式潜意识或连接主义学习。
