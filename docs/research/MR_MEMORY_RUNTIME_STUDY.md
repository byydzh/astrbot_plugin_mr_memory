# 证据契约与反证闭合：真实群聊图记忆的查询时重建、缓存路由与反馈修订

> **状态：研究原型报告，2026-08-17。** 本文报告已经实际调用 Provider 的结果，也明确区分候选召回、潜意识重建、AstrBot 表层回答和盲化自动评分。所有高级语义案例都是反复用于开发的污染案例，`n=1` 结果不作总体外推；线上 0.16 只是不及格弱基线，不是语义金标准。

## 摘要

长期群聊记忆不是静态文档检索：系统既要在严格群作用域和历史 cutoff 内找到证据，还要把改名账号绑定到稳定主体，识别玩笑、讳称和再次引义，在后续反馈到来时修订曾经激活的错误通路，并把延迟控制在聊天可接受范围。现有插件 0.16 虽然让独立潜意识 LLM 在回答前读取宿主预取包，但 54 次线上调用全部止于一次 `host_prefetch`，没有一次进入图工具循环；完成率为 90.74%，已测 Token 合计 1,258,333，墙钟 p95 为 80.35 秒。它因此只能作为运行可观测、语义未达标的弱基线。

本文提出 **ECCR（Evidence-Contract + Counterfactual-Closure Reconstruction，证据契约—反证闭合重建）**。ECCR 不以固定相似度阈值替代语义判断，而让独立 LLM 显式维护查询特定的证据契约：主体、待证义务、竞争释义、必须保留的疑虑、已选路径、未探索前沿和预算。每次检索动作必须绑定某个未闭合义务、可区分竞争假设的判据及预期信息增益；宿主负责群作用域、时间、账号、来源和预算验证，LLM 负责语义 gate。系统只有在义务获得来源支持、反驳、明确争议或安全弃答后才能产生带证据证书的公开简报。查询时临时论证图使用 `SUPPORTS`、`CONTRADICTS`、`IDENTIFIES`、`PRECEDES` 和 `DISCRIMINATES` 等可注册关系；持久图则采用不可变原始证据、双时间版本化派生边、竞争/取代关系和可逆遗忘，避免一次误判覆盖历史。

候选层在 47 条人工批准查询、24,804 条消息和 132 条正例标注上已经可用：Harrier-270M 的 MRR@20、Hit@20、平均 Evidence Recall@20 分别为 0.8413、1.0000、0.6652，优于现在线上 MiniLM 的 0.6411、0.8511、0.5170；但 Harrier 与 BM25 的 MRR 差异区间跨零，embedding 分数也不等于语义质量。在污染案例 #726 上，ECCR v4 用 3 次真实调用、76,224 Token 和 379.394 秒完成两轮检索，访问 40 条、选择 17 条来源，形成了“先称玩吐、后表示购买意向，但不能升级成已付款/抢首发”的校准解释；其 required-group recall 为 1.0、gold-key precision lower bound 为 0.4167。相比之下，旧 B16 为 10,820 Token/21.094 秒，旧 full-MR 为 129,050 Token/73.560 秒。ECCR 的显式反证义务改善了该例的内部简报，但六分钟墙钟足以否决“每次 `/chat` 同步执行完整 ECCR”。表层模型的同一案例 A/B 中，control、B16、ECCR 简报分别消耗 696/2.310 秒、2,419/8.038 秒、2,914/10.110 秒；盲化模型 judge 给出的 composite 为 72/80/100、overall 为 60/80/95。事后逐项证据复核却把 B16 排在 ECCR 前：ECCR 的表层压缩遗漏“玩吐了”和“买”两个核心锚点，模型 judge 又错误声称它明确保留了“抢首发”疑虑。该评分只有一个污染样本、没有人类盲评，而且 judge 首次调用还发生协议失败，不能视为确认性胜利。

两个反例限定了方法边界。“好女孩”案例中，一次普通综合以 24,582 Token/119.303 秒找到了“Bot 表层赞美—群友侮辱性变体—后续螺蛳粉再玩笑化”的核心链，一次 ECCR 以 22,758 Token/92.446 秒反而漏掉该链；第一轮强制审计又因 `removed evidence` 协议错误在 70,033 Token/289.433 秒后失败。宿主修复证据单调性后，第二轮审计以 62,291 Token/215.643 秒完成，找齐“好女孩后紧邻臭表字”和“臭→改口香→好女孩吃香香螺蛳粉”两段事实，却仍只把它们塞进宽泛的 `ironic_meme`，没有建立独立的“侮辱性讳称—二次戏仿”假说；这是证据覆盖成功、假说构建失败。稳定账号/昵称案例则由宿主确定性算法以 0 次模型调用在 2/2 个 cutoff 精确通过；LLM 调用没有带来身份正确性收益。由此本文不声称 ECCR 已胜出，而给出可上线的四层路由：L0 确定性身份与时间约束，L1 带修订向量的证据证书缓存，L2 一次语义读取，L3 仅供高歧义、显式深挖和反馈修订的有界 ECCR。当前证据支持继续实验，不支持把完整重建串行塞进每次聊天，也不支持用缓存彻底取消潜意识语义判断。

**关键词：** 长期对话记忆；图记忆；证据契约；反证闭合；群聊语义；主体绑定；反馈修订；缓存

## 1. 引言

### 1.1 问题背景

传统 RAG 把查询映射到若干文档，再让生成模型根据候选回答 [1]。长期多人群聊却同时包含四种静态文档没有的变化：同一短语会从字面义变成讳称或二次戏仿；同一账号会不断改昵称而重名账号又不能合并；后续成员反馈会推翻早先判断；一次查询所需的证据常分散在相隔数日的事件链中。于是，“找到相似消息”只是第一步，真正的问题是**在当前查询下重建哪条解释路径，并说明它为什么成立、哪里仍不确定**。

该问题还受严格的工程约束。群记忆必须物理隔离，未来消息不能泄漏进历史回答，独立潜意识模型不能拖慢每次 `/chat` 数分钟，图的错误边不能因重复维护而永久强化。若只优化召回，系统会把噪声交给主模型；若只增加 LLM 工具轮数，成本可能上升而语义仍不闭合；若只缓存最终简报，新反馈和别名变化又会静默复用陈旧结论。

### 1.2 研究问题

本文回答以下方向，而不是只复述已有实现：

- **RQ1：真实调用怎样找到并分析记忆？** 候选、图探索、竞争释义、停止条件和最终简报之间应有什么可审计协议？
- **RQ2：实时还是缓存？** 哪些工作可以零调用复用，哪些必须由独立 LLM 在查询时重新判断，如何避免缓存陈旧和实时长尾？
- **RQ3：高级图连接怎样建立？** 多跳、反证、身份、时间和反馈关系如何在查询时形成，又如何安全晋升为持久边？
- **RQ4：代价换来了什么？** Token、墙钟、工具访问和表层回答质量应怎样分账，何时值得进入深层潜意识循环？
- **RQ5：主体与反馈怎样不互相污染？** 数字账号锚点、变化昵称、语义称呼、负面反馈和记忆修订的权限边界在哪里？

### 1.3 本文贡献

本文的新增贡献有四项。

1. 提出并形式化 ECCR：用显式证据义务、竞争释义和反证前沿约束 agentic retrieval，替代“多搜几轮再自由总结”。这是本文的方法提案，不是 MRAgent 原论文已有模块。
2. 给出从群作用域冻结到表层回答的完整双模型流程，并实际运行潜意识 Provider 和表层 Provider；候选召回、重建 brief、最终回答和 judge 成本分开记录。
3. 给出四层实时/缓存路由，以及查询时论证图到持久双时间语义图的晋升、反馈和遗忘生命周期。
4. 保存正结果与负结果：ECCR 在 #726 上能闭合解释，但成本不适合实时；在“好女孩”一次调用上输给普通综合；确定性身份绑定以零调用胜过 LLM。

本文不声称完整复现 MRAgent [2]，不声称 ECCR 已有总体质量优势，也不声称 0.16 的语义理解合格。所有端到端语义结果仍是开发诊断；确认性结论需要未污染 holdout、配对重复和人类盲评。

## 2. 相关工作

### 2.1 RAG、图检索与主动重建

RAG 把参数化模型与外部非参数记忆结合 [1]。GraphRAG 通过实体图与社区摘要处理跨文档全局问题 [3]；HippoRAG 与 HippoRAG 2 使用图传播和检索—推理整合支持多跳记忆 [4,5]。这些方法证明候选排序和关系组织是不同问题，但并不自动解决多人群聊中的说话人、昵称和语用变化。

MRAgent 将记忆表示成 Cue--Tag--Content 图，并让 LLM 在访问时迭代探索、剪枝和重建 [2]。本文继承“记忆是查询时重建而非静态取回”的核心观点，但不照搬自由工具循环。MRAgent 的公开设计没有解决本文所需的持续巩固、遗忘和反馈修订；ECCR 的证据契约、反证闭合及双时间可塑边是本文针对真实线上失败提出的扩展。

### 2.2 动态记忆、时间与反馈

A-MEM 通过类 Zettelkasten 的动态链接使新记忆更新旧记忆的属性和关系 [6]；Zep/Graphiti 将对话组织为时间感知知识图 [7]；Mem0 比较基础记忆与图记忆的质量和成本 [8]。这些工作支持版本化、可追溯图记忆的方向，但生产群聊还需要稳定账号主键、群物理隔离和对玩笑事实的更保守写入。

Reflexion 把自然语言反馈作为 agent 的可复用经验 [9]，MQuAKE 则说明局部知识修改未必能传播到依赖该知识的多跳问题 [10]。本文据此把反馈视为“对实际激活路径的有来源信用分配”，而不是把负面情绪直接当真值或把一句纠正无差别扩散到全图。这一生命周期设计属于本文推论，尚未获得大样本验证。

### 2.3 长期对话评测

LoCoMo 覆盖长程问答、事件总结和多模态对话，并把幽默/讽刺误解和错误说话人归因列为独立错误 [11]。LongMemEval 把长期记忆拆成信息提取、多会话推理、时间推理、知识更新和 abstention，并区分 indexing、retrieval、reading 三阶段 [12]。本文沿用这种分层评测：source overlap 只衡量 retrieval，不能替代 reading 的语义正确性；LLM judge 也不能替代群内语境的人类证据裁决。

## 3. 问题形式化与信任边界

### 3.1 状态与作用域

对群作用域 $s$、请求时间 $t$ 和查询 $q_t$，系统可读状态为：

\[
\mathcal{M}_{s,t}=\{E_{s,<t}, P_{s,<t}, G^d_{s,<t}, G^p_{s,<t}, R_{s,t}\}.
\]

其中 $E$ 是 cutoff 前不可变原始证据，$P$ 是账号主体及时间化别名，$G^d$ 是 Episode/Claim/Topic 等蒸馏图，$G^p$ 是竞争释义和反馈效用组成的可塑图，$R$ 是消息、身份、关系、反馈、提示协议和模型的修订向量。查询期间只构造临时工作图 $A_t$；最终给 AstrBot 表层 LLM 的是有界公开简报 $B_t$，而不是隐藏思维链。

群作用域由当前 AstrBot UMO 决定，不能由用户文本或 LLM 工具参数改写。任何候选、工具结果和 source 展开都必须同时满足同一 scope 与 `sent_at < t`。每个群使用独立数据库；跨群同账号仍不自动共享记忆。

### 3.2 主体层比语义层更可信

主体主键固定为：

\[
\text{Participant}=\text{scope}+\text{platform}+\text{account\_id}.
\]

群昵称、称呼和自称只是带时间的 alias。结构化 mention/reply 可以绑定账号；纯文本新称呼只能提出 alias hypothesis。LLM 可以解释“这句话中的某个称呼可能指谁”，但不能合并账号、改变数字主键或把未来昵称写回历史 cutoff。甲转述乙的偏好必须保存 `speaker=甲, subject=乙`，不能因句法简化写成甲的偏好。

### 3.3 证据层级

系统明确区分四个层级：

1. **候选覆盖**：gold source 是否进入有限候选集合；
2. **路径选择**：模型是否选择真正支持/反驳当前命题的来源；
3. **语义综合**：公开简报是否正确表达命题、竞争释义和保留意见；
4. **表层效用**：AstrBot 主 LLM 最终回答是否更准确、自然且不过度断言。

`required-group recall=1.0` 只代表引用集合覆盖了预注册来源组。`gold-key precision lower bound` 只代表引用 source key 与 gold key 的重合下界。两者都不是 Supported-Claim Precision，更不是人类帮助度。本文所有结果按这四层分别报告。

## 4. ECCR：证据契约—反证闭合重建

### 4.1 设计动机

旧 full-MR 的失败不是“没有工具”，而是工具循环缺少必须解决的问题清单：模型可以不断扩张相关主题，却不必说明哪条新证据改变了哪个判断。固定相似度阈值也无法解决这个问题，因为 embedding 只是节点距离先验，不知道一句话是事实、反话还是对旧梗的再引用。

ECCR 将每次重建视为一个有界证明搜索。LLM 可以自由判断语义和注册局部关系，但必须在宿主可验证的证据契约内行动。形式上，第 $k$ 轮状态为：

\[
C_k=(S,Q,T,V,\mathcal O_k,\mathcal H_k,\mathcal U_k,
\mathcal P_k,\mathcal F_k,\mathcal A_k,B_k),
\]

其中 $S/Q/T/V$ 分别是 scope、query、cutoff 和修订向量；$\mathcal O$ 是证据义务；$\mathcal H$ 是竞争解释；$\mathcal U$ 是必须保留的不确定性；$\mathcal P$ 是已经选择的有来源路径；$\mathcal F$ 是未探索前沿；$\mathcal A$ 是已经尝试的动作；$B$ 是调用、Token、墙钟和工具预算。

每个义务只能处于 `OPEN`、`SUPPORTED`、`REFUTED`、`CONTESTED`、`AMBIGUOUS` 或 `EXHAUSTED`。至少一个关键义务存在；任何终态的歧义、耗尽或保留意见都必须在公开简报中可见。模型不能用“我觉得足够了”绕过未闭合关键义务。

### 4.2 真实调用的逐步流程

一次真实 `/chat` 的记忆路径如下。

1. **宿主冻结。** 由事件确定 scope、cutoff、当前账号/别名快照和修订向量；未来消息与跨群来源在进入 LLM 前即被排除。
2. **混合候选初始化。** 本地 embedding、BM25、时间邻域、reply/mention 和现有图锚点并行生成候选。分数只排序，不作语义终判。
3. **第一次潜意识调用。** 独立 LLM 读取查询、最近上下文和候选，列出主体、关键义务、竞争释义、疑虑以及需要的下一步动作。第一次调用不能直接把未检查前沿伪装成终局。
4. **动作绑定。** 每个动作必须给出 `obligation_id`、要区分的竞争解释、预期信息增益和有界参数。例如，搜索“买”不是因为词面相似，而是为了区分“表达购买意向”和“已经付款/抢首发”。
5. **宿主执行与审计。** 只读工具验证 scope、cutoff、source allowlist、主体和预算后返回原始来源、事件邻域或图桥；未知来源、未来消息和账号改写被拒绝。
6. **契约更新。** LLM 只能依据新增 source 或经验证的图桥改变义务状态。每次状态转移记录 result hash、身份变化、前沿变化和实际增益；重复同一来源不算新证据。
7. **反证闭合。** 下一轮必须优先检查最可能推翻当前解释的反例、未覆盖事件和竞争释义，而不是继续收集同方向材料。默认最多两轮检索、三次 LLM 调用。
8. **证书与表层交接。** 当关键义务均闭合、反证前沿耗尽或系统安全弃答时，输出包含 claims、conflicts、unresolved、selected sources 和 stop reason 的证据证书，再压缩为表层简报。AstrBot 主 LLM 仍决定最终措辞和是否调用 Web Search；潜意识层不接管主 Agent。

停止条件不是单一阈值，而是下列任一可审计终态：`CERTIFIED_CLOSE`、`SAFETY_ABSTAIN`、`FRONTIER_EXHAUSTED`、`SATURATED` 或 `BUDGET_EXHAUSTED`。若预算耗尽但关键义务未闭合，系统只能返回显式 partial/abstain，不能把诊断草稿伪装成有效简报。

### 4.3 查询时高级论证图

ECCR 在每次查询中建立临时论证图，而不是直接污染长期图。节点包括主体、事件、主张、竞争释义、反馈和当前问题；关系类型可以注册，但必须声明语义方向和来源约束。最小关系集为：

- `SUPPORTS(e,h)`：证据支持解释；
- `CONTRADICTS(e,h)`：证据反驳或降低解释；
- `IDENTIFIES(e,p)`：结构化证据把称呼绑定到账号主体；
- `PRECEDES(e_i,e_j)`：建立时序而不暗示因果；
- `DISCRIMINATES(e,h_i,h_j)`：证据能区分两个竞争解释；
- `SUPERSEDES(h_j,h_i)`：后续有来源解释取代旧解释但保留历史。

关系名可扩展不等于 LLM 可以任意写边。宿主验证端点 scope、cutoff、来源和账号约束；LLM 决定的是“这条来源在当前问题中支持哪种语义”。这样 embedding 继续充当距离工具，LLM 才是语义 gate，同时保留可重放证据。

### 4.4 从临时图到持久图的生命周期

查询时路径只有满足重复支持、跨事件稳定或明确人类反馈时，才可提交为持久边 proposal。持久层采用以下规则：

1. 原始消息、reply、mention 和账号锚点不可变；派生边另存而不覆盖证据。
2. 每条派生边同时记录 valid time 与 transaction time、source group、prompt/model/protocol revision、状态和 supersedes 链。
3. `epistemic_confidence` 与 `activity_utility` 分离。前者表示命题证据，后者表示该路径对近期回答的帮助；“常用”不能提高事实真值。
4. 重复写入同一 source group 幂等，不能通过重跑维护虚增 support 或 utility。
5. 遗忘首先使低效用派生边休眠；历史 revision 与原始证据仍可回放。合并必须可逆并保留竞争解释。
6. 查询只读取 cutoff 对应的版本快照，不能用当前状态解释历史问题。

这套持久化设计目前是工程目标而非已完成能力。现有实现仍有历史 cutoff 读取当前派生状态、缺少完整 revision vector、utility decay 可能使休眠边异常复活、同源 upsert 可能累加支持、关系修订可能批量改写以及文本别名歧义 fallback 等阻断项；在这些问题解决前，只读查询时 ECCR 可以实验，自动持久图修改不能宣告生产就绪。

系统不保存潜意识 LLM 的完整隐藏状态。临时论证图在请求结束后丢弃，只持久化有来源证书、必要派生边及其 revision，因此存储增长约为原始证据加被晋升边的版本数，而不是“每次思考的全部 token”。图片正文同样不持久化；重复媒体只允许有界 opaque hash 统计。性能瓶颈由本地索引和 Provider 调用分开计量，不能把图数据库字节数与 LLM 上下文成本混为一谈。

### 4.5 反馈修订

反馈不是对整张图做“正向/负向强化”，而是对**本次回答实际激活的路径**做有来源的信用分配：

```text
request → selected evidence/path → surface answer → later human feedback
        → feedback hypothesis → counterexample audit → versioned mutation
```

弱负面情绪只把相关边转为 `CONTESTED` 并打开复核义务；明确纠正、重复一致反馈或管理员确认才能 `SUPERSEDE`/休眠高影响边。反馈不能修改无关路径、原始证据或账号主键。若“元素太密集”是对一次生图的后续反馈，系统应把“该用户/群在该场景偏好更疏的构图”作为可撤销行为假说，而不是把“密集构图永远错误”写成事实。

## 5. 实时与缓存：四层路由

实时和缓存不是二选一。本文根据语义风险与延迟设计四层路由。

| 层 | 工作 | LLM 调用 | 适用条件 | 主要失效模式 |
| --- | --- | ---: | --- | --- |
| **L0 确定性宿主层** | scope、cutoff、账号/别名、reply/mention、撤回/修订验证 | 0 | 每次请求必经 | 实现 bug，而非语义不足 |
| **L1 证据证书缓存** | 混合候选、source 展开、局部图桥、上次证明包 | 0 | 修订向量完全命中 | 陈旧 revision、查询语境变化 |
| **L2 一次语义读取** | LLM 对新查询选择证据、竞争释义和保留意见 | 1 | 普通记忆问题、证据局部完整 | 一次调用过早收敛 |
| **L3 有界 ECCR** | 反例/多跳/主体歧义/反馈闭合 | 2–3 | 显式深挖、高风险、L2 未闭合 | 分钟级延迟、上下文膨胀 |

L1 缓存的是**带证明的证据状态**，不是可以无条件复用的最终语义答案。cache key 至少包含：

\[
K=(scope,query/context,cutoff,message,graph,identity,relation,
feedback,protocol,model\ revisions).
\]

新消息、别名变化、关系修订、反馈 proposal 或模型/提示版本变化都使相关证书失效。主请求可与 L2/L3 并发：若记忆证书在短 deadline 内完成，就注入本次表层回答；否则主回答使用最近仍有效的证书，并让深层分析在后台完成，供后续轮次或明确追问使用。这里的关键不是“为了快而不调用”，而是把六分钟级完整闭合移出每次同步关键路径，同时保留潜意识功能。

L2 的实际 DSV 调用仍有 25–119 秒级单例延迟，因此当前 Provider 下也未达到每次聊天同步可接受水平。要上线实时语义，不应通过关闭思考来假装解决，而应实测更快 Provider、流式/并发 deadline、预计算证书和 L3 触发率；质量门和延迟门必须同时通过。

## 6. 实现

### 6.1 双模型与可审计账本

插件的潜意识 Provider 与 AstrBot 主 LLM 分离。实验中的潜意识调用使用已配置的 DeepSeek V4 Flash；表层 A/B 使用 AstrBot 已配置的 Gemini 3.5 Flash。每一阶段分别记录 ordinary/cached/output Token、墙钟、工具动作、source key、公开 brief、结果 hash 和停止原因；隐藏思维链不保存。usage 未知时按失败关闭，不能把未知成本记为零成功。

结构化协议采用严格 schema。解析器尝试所有可见结构化候选，但若都不合格，保留最接近完成输出的真实 schema 错误；不会再让 reasoning fallback 的次级错误遮蔽 completion 错误。所有超时是墙钟硬限制，工具和模型预算独立计数。

## 7. 实验设计

### 7.1 数据与案例

本文使用四类材料。

1. **候选检索集：** 47 条人工批准查询、24,804 条 cutoff 前消息、132 条正例证据标注。
2. **线上 B0 观察窗：** 单群 54 次真实调用；只有运行状态和成本，没有语义标签。
3. **#726 严格遮罩案例：** 固定真实 query、cutoff、候选和隔离图，输入 packet hash 绑定；该案例已反复调试，属于污染开发集。
4. **机制回归案例：** “好女孩”包含 73 条真实来源和 8 个时间 episode；昵称案例跨四个昵称、约 125 天并设置两个历史 cutoff。它们只验证特定机制，不作总体估计。“好女孩”packet 由冻结来源离线组成并把全部 episode 直接提供给模型，因此测的是 synthesis/controller，不是线上检索器；不能把模型读取 oracle packet 的结果记成 retrieval recall。

真实正文、账号、群号和数据库保存在 Git ignored `.dev/`；可提交材料只含脱敏聚合。所有候选与图访问必须早于 cutoff。

### 7.2 评价指标

候选层报告 MRR@20、Hit@20、Evidence Recall@20 和 All-Evidence@20。端到端层分别报告：required-group recall、gold-key precision lower bound、Supported-Claim Precision、人类帮助度、false-none、正确保留疑虑、Unsupported Claims、调用数、Token 和墙钟。前两个来源指标不能替代后三个语义指标。

候选差值使用查询为配对单元的 20,000 次 percentile bootstrap（seed `20260817`）。Provider 案例目前每 arm 只有一次，不计算置信区间。表层 judge 对 arm 标签盲化，但仍是模型 judge；必须与独立人类盲评分开报告。

### 7.3 #726 的分阶段协议

旧基线包含 A1/cache、B16 单次 gate 和 full-MR 自由工具循环。ECCR 经过四次工程迭代：v1 在首次调用后触达预算而未检索；v2/v3 暴露结构化协议错误；修复解析器与安全降级后，v4 才完成两轮检索。最终表层 A/B 固定同一最近上下文，只替换记忆 brief；control 不提供长期记忆。

### 7.4 “好女孩”反证测试与主体测试

“好女孩”的目标不是列举主题，而是比较至少四种解释：表层褒义、作品/表情包来源、侮辱性讳称、群友对该讳称的二次玩笑化。正确答案必须区分 Bot 说法、人类说法和研究者推断，并保留“邻接不等于明确 reply”的疑虑。

昵称测试把确定性宿主算法与 LLM 解释分开。成功条件是两个 cutoff 都使用当时已经出现的别名、排除未来别名、保持同一数字账号且不跨 scope。LLM 可以补充称呼语义，但不能成为账号同一性的必要条件。

## 8. 结果

### 8.1 候选召回可用，但不能代替语义 gate

**表 1　人工批准群聊检索集（47 queries / 24,804 documents / 132 positives）**

| 检索器 | MRR@20 | Hit@20 | Mean Evidence Recall@20 | All-Evidence@20 |
| --- | ---: | ---: | ---: | ---: |
| MiniLM-L12-v2 | 0.6411 | 0.8511 | 0.5170 | 0.2553 |
| BM25 | 0.8357 | 0.8936 | 0.5649 | 0.2340 |
| Harrier-270M | **0.8413** | **1.0000** | **0.6652** | **0.3617** |
| Harrier + `group-memory` prefix | 0.7632 | 0.9787 | 0.6617 | 0.3830 |

**表 2　Harrier 配对差值与 95% percentile bootstrap 区间**

| 比较 | 指标 | 差值 | 95% 区间 |
| --- | --- | ---: | ---: |
| Harrier − MiniLM | MRR@20 | +0.2002 | [+0.0752, +0.3271] |
| Harrier − MiniLM | Evidence Recall@20 | +0.1482 | [+0.0596, +0.2376] |
| Harrier − BM25 | MRR@20 | +0.0056 | [−0.0990, +0.1120] |
| Harrier − BM25 | Evidence Recall@20 | +0.1004 | [+0.0142, +0.1933] |
| Harrier prefix − default | MRR@20 | −0.0782 | [−0.1398, −0.0271] |
| Harrier prefix − default | Evidence Recall@20 | −0.0035 | [−0.0638, +0.0461] |

Harrier 相对 MiniLM 的优势明确，但相对 BM25 的 MRR 区间跨零；人工加任务前缀还显著降低 MRR。工程结论是保留 embedding 与词面信号并做融合，而不是把较新的模型、前缀或向量距离包装成语义真值。

### 8.2 线上 0.16 是高成本、未验证的不及格弱基线

**表 3　B0 单群观察窗口**

| 指标 | 结果 |
| --- | ---: |
| 调用数 | 54 |
| 完成 / 失败 | 49 / 5 |
| brief / none / failed | 25 / 24 / 5 |
| `fast` / 仅 `host_prefetch` | 54 / 54 |
| escalation / 图工具循环 | 0 / 54 |
| 错误 | 3 × JSON `ValueError`; 2 × timeout |

**表 4　B0 每次调用的已测成本**

| 分布 | Mean | Median / p50 | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Ledger Token / run（n=54；2 条缺失记测得 0） | 23,302 | 21,315.5 | 43,086.7 | 46,854 |
| Usage-present Token / run（n=52） | 24,198.7 | 21,450.5 | 43,220.9 | 46,854 |
| Wall latency / run | 20.61 s | 7.00 s | 80.35 s | 90.00 s |

54 次合计记录 60 个 Provider 调用、1,258,333 Token，其中 ordinary input 1,013,599、cached input 136,576、Provider-reported output 108,158。两次 timeout 在 usage 返回前结束，因此总量只是已测成本，不是账单上界。更重要的是，这些调用没有人工语义标签；“brief”不等于正确，“none”也不等于正确 abstention。本文称其“不及格”是工程状态判断：主动重建 0/54、失败率 9.26%、语义质量未被证明，而不是虚构一个准确率分数。

### 8.3 #726：ECCR 能闭合案例，但成本和样本量都不支持宣告胜利

该调用询问一位成员是否对某类游戏“口嫌体正直”。证据链包括其先前“类魂玩吐了”、后来询问相关新游戏并说“买”，以及其他群友把游戏描述为类魂 PvE/PvP。合理简报应区分“表达购买意向”与“已经付款/预购/抢首发”，并说明“口嫌体正直”是观察者概括而非当事人自称。

**表 5　同一污染开发调用的重建 arms**

| Arm | Calls / tools | Visited / cited | Required-group recall | Gold-key precision LB | Token | 墙钟 | 诊断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 local materialization | 0 / 0 | 38 / 38 | 1.0 | 0.1579 | 0 | 0.001277 s | 材料堆叠，无语义终判 |
| B16 single gate | 1 / 0 | 38 / 6 | 1.0 | 1.0000 | 10,820 | 21.094 s | 简洁且校准，但漏后续游玩证据 |
| full-MR | 9 / 29 | 62 / 10 | 1.0 | 0.7000 | 129,050 | 73.560 s | 找到更多证据，最终未闭合目标释义 |
| **ECCR v4** | **3 / 6** | **40 / 12** | **1.0** | **0.4167** | **76,224** | **379.394 s** | 闭合时序冲突并保留购买措辞疑虑 |

旧三臂合计 10 次 Provider 调用、139,870 个已测 Token。ECCR v4 的三次调用分别为 23,251 Token/105.444 秒、21,267/100.932 秒、31,706/172.998 秒。它在两轮中共执行 associations、event context/keywords、media patterns、personal aspect 和 tag events 等 6 次有界检索，最终 `CERTIFIED_CLOSE`：承认“玩吐了”与后来“买”的张力，同时拒绝把“买”升级成直接付款或抢首发证据。这个公开 brief 的校准性是实际方法收益，但 379 秒也是真实代价。

**表 6　ECCR 工程迭代不是可丢弃的零成本失败**

| 版本 | 调用 | Token | 墙钟 | 结果 |
| --- | ---: | ---: | ---: | --- |
| v1 | 1 | 28,564 | 156.706 s | 首次调用后预算耗尽；0 轮检索、无 brief |
| v2 | 1 | 23,314 | 98.606 s | 结构化协议失败 |
| v3 | 1 | 23,005 | 111.485 s | 实际 completion schema 错误曾被 fallback 错误遮蔽 |
| v4 | 3 | 76,224 | 379.394 s | 两轮检索后闭合 |

v4 的 required-group recall 与所有旧 arm 一样都是 1.0，因此不能用该数值证明 ECCR 更懂语义；其 gold-key precision lower bound 也低于 B16。真正的差异在公开简报是否逐项裁决冲突，但目前只有研究者复核而没有独立人类盲评。

### 8.4 表层回答 A/B 证明“brief 不是答案”，但仍只是 n=1 自动评测

固定同一真实 query 和最近上下文后，AstrBot 表层模型分别收到无长期记忆、B16 brief 和 ECCR brief。

**表 7　#726 表层生成与盲化模型评分**

| Arm | 表层生成 Token | 表层墙钟 | Blind-model composite | Blind-model overall |
| --- | ---: | ---: | ---: | ---: |
| Control | 696 | 2.310 s | 72 | 60 |
| B16 brief | 2,419 | 8.038 s | 80 | 80 |
| ECCR brief | 2,914 | 10.110 s | 100 | 95 |

自动 judge 的排序不能通过逐项证据复核。Control 只给出泛化“真香”调侃；B16 明确恢复“类魂玩吐了”、开服期游玩及“抢首发不确定”，但漏掉“买”和 PvEvP 归类；ECCR 恢复 PvEvP、开服期游玩与多重动机，却在表层压缩中丢掉“类魂玩吐了”和“买”，也没有直接说明“抢首发/预购无原话证据”。因此事后规则复核排序为 **B16 > ECCR > Control**，而不是自动 judge 的 ECCR > B16 > Control。该不一致本身回答了一个关键问题：更好的内部图或 brief 不保证更好的最终回答，reading/surface compression 必须作为独立层评测。

样本已经被多轮开发污染，独立样本数为 1，judge 是模型而非人类，且首次 judge 调用还因输出协议失败消耗 1,840 Token/5.068 秒，重试再消耗 1,793/4.782 秒。三次生成与两次 judge 共 5 次完成调用、9,662 Token 和 30.308 秒。盲化只隐藏 arm 标签，不能消除 judge 偏差或同模型族偏差；两个分开的重试账本还复用了同一 request ID，跨账本汇总必须以 ledger hash 与 request ID 联合去重。因此 100/95 只是应被保留的原始模型评分，不是 ECCR 胜率或准确率。

### 8.5 “好女孩”：普通一次综合找到核心链，一次 ECCR 反而漏掉

在 73 条来源、8 个 episode 的固定 oracle packet 上，一次普通综合用 24,582 Token/119.303 秒找到了以下核心链：Bot 对某人的“好女孩”表层赞美；紧邻的人类侮辱性变体；后来围绕螺蛳粉“臭/香”的讨论及“好女孩就应该吃香香螺蛳粉”的再次引用。它同时保留“没有显式 reply，是否刻意再引义仍未证实”的疑虑。全部 episode 已被直接提供，这一结果只能评价模型综合和控制协议，不能证明真实数据库检索会找到它们。

一次 ECCR 用 22,758 Token/92.446 秒覆盖多个表层释义和螺蛳粉语境，却没有把侮辱性变体选入核心路径。两者自动 required-group recall 都是 1.0，正好证明旧 gold group 过宽、source overlap 会掩盖语义失败。强制两轮 counterexample audit 的第一次运行进行了 2 次真实调用、70,033 Token、289.433 秒，却因第二轮错误移除第一轮已选证据而触发 `removed evidence` 协议失败；这是控制器状态复制缺陷，不是语义“不存在”。

| Arm | Calls | Visited sources | Token | 墙钟 | 核心侮辱—再玩笑链 |
| --- | ---: | ---: | ---: | ---: | --- |
| One-pass synthesis | 1 | 73 | 24,582 | 119.303 s | 找到；保留“无显式 reply”疑虑 |
| One-call ECCR | 1 | 26 | 22,758 | 92.446 s | 漏掉 |
| Forced two-call audit v1 | 2 | 73 | 70,033 | 289.433 s | `removed evidence`，无有效 brief |
| Forced two-call audit v2 | 2 | 73 | 62,291 | 215.643 s | 找齐两段事实，未建立核心语义桥 |

修复版由宿主只做“上一轮同 ID 条目的证据并集恢复”，不允许模型借机换状态、主体或来源；本次只恢复了 `sincere_praise` 上一条较弱 support，并没有替模型补核心语义。v2 最终把 literal praise 与 broad `ironic_meme` 标为 `CONTESTED`，明确机器人文本不能作为人类语义或 MyGO 梗史的独立真值，并分别重建“好女孩→臭表字”与螺蛳粉“臭→香→好女孩”的时间线。然而 contract 从头到尾只有 `sincere_praise`、`ironic_meme`、`assistant_flattery_template` 三个宽候选，未建立冻结 gold 所要求的独立 `insult-euphemism-and-second-order-rejoke` 假说，也没有专门保存“二次戏仿连接尚未证实”的 uncertainty。自动 recall 仍为 1.0，人工语义结论只能是**部分成功**。

两次 audit 合计 4 次调用、132,324 Token 和 505.076 秒；四种“好女孩”真实运行总计 179,664 Token/716.825 秒。成本没有换来核心测试通过。当前结论是：ECCR 的 contract 形式本身不保证反例被发现；第一轮 compile 若遗漏真正判别维度，第二轮不能只做 frozen-schema coverage audit。下一版应允许第二轮添加由未选证据直接约束的 `AUDIT_DISCOVERY` 竞争假说：旧条目不可删除，新条目必须引用至少一个此前未选来源、声明它区分的旧候选，并在同轮以 `HYPOTHESIS/CONTESTED + unresolved` 收口；它不能直接晋升持久图真值。该案例在核心假说构建上仍未通过。

### 8.6 主体确认应由确定性宿主完成，LLM 只做称呼语义

在同一账号跨四个昵称、两个历史 cutoff 的测试中，宿主确定性算法以 0 次 Provider 调用精确通过 2/2 cutoff：使用当时最新可见昵称，排除未来昵称并阻止跨 scope 来源。一次普通 LLM 综合额外消耗 7,213 Token/25.564 秒，内容语义正确但未输出结构化 `subjects`；一次 ECCR 消耗 22,064 Token/119.214 秒并显式绑定主体，结果也正确。

这不是 ECCR 的身份优势。既然零调用宿主已经得到相同账号真值，LLM 的 7k–22k Token 和 25–119 秒没有增加确定性正确率。正确分工是 L0 保证账号同一性，L2/L3 只解释“这段自然语言称呼是否在当前语境中指向该账号”并在歧义时 abstain。旧测试脚本的 snapshot 实现问题已修复，不作为算法结论。

### 8.7 反馈修订只有局部行为证据

现有反馈 A/B 只有两个手选行为样本。`no_followup` 的 control 和 memory 都是 3/3，存在天花板；`forced_choice` 中 control 为 0/3、memory 为 3/3，说明 sender-scoped 行为规则能在一个案例中跨词面生效。一次维护与激活共 12,310 Token，最终回答 arms 共 16,995 Token。

这不能证明通用记忆修订：两个样本共享底层调用，评分只看指定行为，没有测试事实性、错误反馈抵抗或无关任务保持；历史 166 条维护决策中还没有一次显式 LLM 前向图修改。现阶段只能说反馈协议和单例激活可运行。

## 9. 代价、收益与工程决策

### 9.1 生命周期必须分账

一次性历史导入、历史图构建、日常增量整理、每请求语义重建、反馈维护、本地 embedding 和表层新增输入是不同成本中心。一次性回填不能挤占日常在线预算，也不能因为回填量大而让插件每五分钟重复处理同一待整理队列。

在线收益必须以最终质量增量为分母：

\[
C_{helpful}=\frac{T_{memory}+T_{surface\ increment}}
{\#\text{相对 control 获得人类盲评质量增益的请求}}.
\]

同样报告 $\Delta Q/\Delta Token$ 与 $\Delta Q/\Delta latency$。没有人类盲评时，只能报告成本和诊断，不能声称“每 Token 更值”。

### 9.2 当前成本—收益结论

- **B0：** 平均已测约 23k Token，p95 80.35 秒，却从未进入工具循环；语义收益未知。
- **B16：** #726 上最便宜的有语义 brief，21.094 秒；遗漏一部分跨事件证据。
- **full-MR：** 129,050 Token 扩大证据面但未闭合语义，说明自由多轮浏览的边际收益很差。
- **ECCR：** 76,224 Token 少于 full-MR，但墙钟 379.394 秒远高于所有旧 arm；该例 brief 更完整，仍不足以支撑同步普及。
- **确定性身份：** 0 调用获得 2/2 cutoff exact，是应无条件前置的便宜约束。
- **“好女孩”：** 更复杂协议没有稳定超过普通综合，说明成本不能作为能力代理。

因此合理的产品策略不是砍掉潜意识，而是让绝大多数请求停在 L0/L1，普通语义请求进入一次 L2，只有显式深挖、竞争释义、反馈或高影响不确定性进入 L3。L3 应异步或在用户明确等待时运行；若未来更快 Provider 把 L2 压到聊天 deadline 内，再依据真实质量—延迟 Pareto frontier 调整触发率。

### 9.3 上线门槛

ECCR 从研究原型进入默认在线路径前至少需要：

1. 未污染、跨群 holdout 上的配对人类盲评，且 Supported-Claim Precision、false-none 和正确疑虑保留优于 B0/B16；
2. L2 的 p95 达到产品 deadline，L3 有异步状态和可见进度，不阻塞普通 `/chat`；
3. cache revision vector、cutoff 回放和同源幂等通过故障注入；
4. 结构化解析和 unknown usage 全部 fail closed；遗漏既有 provenance 由宿主单调恢复并审计，语义状态突变仍拒绝；所有重试单独计费；
5. 持久边的双时间读取、supersedes、休眠/恢复和管理员回滚可审计；
6. 身份主键永远由宿主控制，LLM alias proposal 不能自动跨账号或跨群合并。

在这些门槛前，可上线的是只读证据证书、调试控制台和手动深挖，不是自动修改全状态图。

## 10. 讨论

### 10.1 ECCR 新在哪里，又没有解决什么

ECCR 相对旧 full-MR 的核心增量不是更多工具，而是将“为什么继续搜、什么证据会推翻我、何时可以停”变成可验证状态。#726 v4 证明这一控制结构可以形成比自由浏览更校准的 brief；“好女孩”又证明第一次 contract 仍可能遗漏真正的判别维度。第二轮虽然看到了完整未选 evidence，却因 schema 冻结只能把新线索塞进既有宽标签，出现“证据覆盖成功、假说构建失败”。方法下一步不是继续放宽 Token，而是把 audit 分成两种状态迁移：旧义务/证据保持单调；新发现只能创建带 `AUDIT_DISCOVERY` provenance 的候选假说，必须引用此前未选来源、说明其区分的旧解释，并以 `HYPOTHESIS/CONTESTED` 和显式 unresolved 收口，禁止同轮晋升为持久真值。

### 10.2 高级语义连接不是预设状态机，也不能完全无约束

动态关系注册和 LLM gate 允许模型构造新语义，如“表层赞美—侮辱性讳称—二次友好戏仿”的竞争网络；宿主状态机只约束 scope、证据和版本，不预先规定最终含义。这避免用前 AI 时代的死阈值替代语义，同时也避免让 LLM 通过幻觉边重写账号或历史。换言之，连接主义式活动传播可以发生在临时论证图里，长期学习则通过有来源、可回滚的版本边实现。

### 10.3 实时能力的正确含义

“不在普通 `/chat` 串行跑六分钟”不等于取消潜意识。实时系统可并行启动 L2，命中 deadline 就注入本轮；未命中时保存证书供下一轮，并在明确追问时进入 L3。真正需要实测的是：多少请求在 L1 已足够、L2 增益有多大、L3 的触发率和完成时间，而不是用固定 4k/8k 输出上限或思考 0 强行制造低延迟。

### 10.4 为什么仍保留 BM25 和确定性逻辑

使用现代 LLM 不意味着每一层都应由 LLM 搬运。BM25 在该检索集上与 Harrier 的 MRR 几乎相同，数字账号在两个 cutoff 上零调用精确。让 LLM 处理可确定问题只增加延迟和新的失败面。更“现代”的系统是把确定性真值、统计候选和语义判断放在各自最合适的位置，而不是让单一模型垄断控制。

## 11. 有效性威胁

### 11.1 内部与统计有效性

Provider 排队、模型版本、缓存预热和后台任务会影响时延。#726、“好女孩”和昵称实验每 arm 只有一次；单次重复不能扩大独立样本数，也不能计算总体显著性。表层 judge 第一次还发生协议失败。所有案例结果只能定位机制和失败面。

### 11.2 构念有效性

旧“好女孩”rubric 只测主题聚合，曾把未识别核心语义的结果评为高分。新的 gold 必须在看 arm 输出前冻结竞争释义、必要证据、允许保留和禁止外推。source overlap、引用数量、图连通性和模型自报置信度均不能替代语义裁决。

### 11.3 开发污染与 judge 偏差

#726 和“好女孩”已被多轮 prompt 调试，不能进入确认集。模型 judge 即使看不到 arm 标签，也可能偏好更长、更结构化或同模型风格的回答。确认性实验需要独立人类按证据盲评，调用或群作为 cluster，并披露排除和重试。

### 11.4 时间泄漏与持久状态

只过滤原始消息时间不够；topic、claim、alias、反馈和派生边的 revision 都可能把未来状态带回历史。当前 hash-bound masked packet 能检测输入漂移，但若隔离图不是从冻结消息可重放构建，仍依赖研究者 attestation。正式实验必须保存输入 hash、构建协议、派生日志和产物 hash。

### 11.5 外部有效性

47 条检索题和机制案例来自少数群，无法外推到其他社区、私聊、企业知识库或视觉表情语义。本文也不持久化图片；高频媒体最多以 opaque hash 做频率候选，多模态语义成本和隐私需另立实验。

## 12. 结论

本文提出并实际测试了 ECCR，而不是把“完整重建”停留在架构口号。它把真实调用拆成宿主冻结、混合候选、证据契约、动作绑定、反证闭合、证据证书和表层回答七个可审计阶段；同时给出 L0 确定性宿主、L1 证书缓存、L2 一次语义读取、L3 有界深挖的产品路由，以及查询时论证图向双时间持久图晋升和反馈修订的工程方案。

现有数据给出的结论是克制而明确的。候选层已经比 MiniLM 好，但 embedding 不是语义终判。0.16 在 54 次线上调用中从未主动重建，是高成本、质量未验证的不及格弱基线。ECCR 在 #726 上以 3 次调用闭合了真实时序冲突，但 76,224 Token 和 379 秒禁止其成为每次 `/chat` 的同步默认路径；表层模型随后又丢掉简报中的核心锚点，自动 judge 还把这种覆盖损失误判为满分。“好女孩”两轮审计找齐两段事实却没有构造核心语义桥，进一步证明 evidence coverage、hypothesis construction 与 final reading 必须分开评测。账号主体则应由零调用确定性逻辑绑定，LLM 只负责称呼解释。

因此当前方向不是取消潜意识，也不是无条件释放深层循环：先修复反证审计和版本化图的不变量，用未污染人类盲评验证 L2/L3 的真实增益，再依据质量—延迟 Pareto frontier决定默认路由。在那之前，ECCR 是可运行、可审计、尚未证明总体优越的研究原型。

## 参考文献

1. Lewis, P. et al. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS, 2020.
2. Ji, S., Li, Y., Hooi, B. [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036). arXiv, 2026. [Official implementation](https://github.com/Ji-shuo/MRAgent).
3. Edge, D. et al. [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). arXiv, 2024.
4. Gutiérrez, B. J. et al. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831). arXiv, 2024.
5. Gutiérrez, B. J. et al. [HippoRAG 2: From RAG to Memory](https://arxiv.org/abs/2502.14802). arXiv, 2025.
6. Xu, W. et al. [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110). arXiv, 2025.
7. Rasmussen, P. et al. [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956). arXiv, 2025.
8. Chhikara, P. et al. [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413). arXiv, 2025.
9. Shinn, N. et al. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). NeurIPS, 2023.
10. Zhong, Z. et al. [MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions](https://arxiv.org/abs/2305.14795). EMNLP, 2023.
11. Maharana, A. et al. [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753). arXiv, 2024.
12. Wu, D. et al. [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813). ICLR, 2025.

## 附录 A：指标定义与误读边界

| 指标 | 定义 | 不能证明 |
| --- | --- | --- |
| MRR@20 | 第一条 gold evidence 在前 20 的倒数排名均值 | 其余必要证据已找全 |
| Hit@20 | 至少一条 gold evidence 进入前 20 的比例 | 多跳问题已可回答 |
| Evidence Recall@20 | 每题前 20 命中数 / gold 数，再求均值 | 候选没有噪声 |
| Required-group recall | 每个预注册来源组至少引用一条 | 语义解释正确 |
| Gold-key precision LB | 引用 key 中属于 gold key 的比例下界 | claim 被来源蕴含 |
| Supported-Claim Precision | 可验证 claim 中有本轮来源支持的比例 | 回答自然或对用户有帮助 |
| False None | 有必要记忆却返回 none 的比例 | 需要独立人工 gold |
| Correction Specificity | 修订后无关查询保持正确的比例 | 只测目标问题不能得到 |

## 附录 B：预注册 arm 与控制变量

正式配对实验至少包含：C0 最近上下文、R0 raw top-k、B0 0.16 弱基线、A1 证书缓存、L2 一次语义读取、L3/ECCR 有界闭合。所有 arm 固定同一 query、cutoff、群、隔离图、主 LLM、system prompt、最近上下文、gold、重试政策和表层生成参数；唯一改变的是记忆路径。执行顺序交替/随机，失败重试单独计费。

正式结果必须同时报告 Evidence Recall、Supported-Claim Precision、False None、Unsupported Claims、正确保留疑虑、人类帮助度、Token/run、首块与完整 p50/p95。A2/L3 可以自行停止，但未闭合关键义务不得伪装成完整 brief。

## 附录 C：主体与高级链接验收矩阵

| 场景 | 宿主确定性行为 | LLM 允许行为 | 禁止行为 |
| --- | --- | --- | --- |
| 同账号改昵称 | 同一 Participant；保存时间化别名 | 解释昵称语境 | 新建第二人物 |
| 两账号同昵称 | 保持两个主体并提示歧义 | 提出两个称呼候选 | 合并账号 |
| mention/reply 新称呼 | 绑定结构化目标账号 | 提交 alias hypothesis | 显示名覆盖主键 |
| 纯文本新昵称 | 唯一命中才作候选，否则 unresolved | 请求确认 | 无证据自动绑定 |
| 甲转述乙偏好 | `speaker=甲, subject=乙` | 标注转述与不确定 | 写成甲的偏好 |
| 玩笑身份说法 | 原证据保留，高风险 claim 隔离 | 竞争释义 | 写成确定身份 |
| 后续负面反馈 | 只打开实际激活路径复核 | 提交 supersede proposal | 改无关边或原始证据 |

## 附录 D：复现入口与隐私边界

- [论文机制覆盖矩阵](../PAPER_COVERAGE.md)
- [系统架构](../ARCHITECTURE.md)
- [反馈闭环](../FEEDBACK_LOOP.md)
- [主体模型](../IDENTITY_MODEL.md)
- [检索测试集协议](../RETRIEVAL_BENCHMARK.md)
- [开发与实验记录](../DEVELOPMENT_LOG.md)
- [隐私安全聚合](data/observed_evidence.json)
- [旧三臂 Provider 聚合](data/pilot_provider_smoke_aggregate.json)
- [聚合分析脚本](../../scripts/analyze_runtime_study.py)
- [遮罩重建脚本](../../scripts/masked_ab_experiment.py)
- [固定 packet / ECCR 审计脚本](../../scripts/eccr_packet_experiment.py)
- [表层回答 A/B 脚本](../../scripts/surface_brief_ab_experiment.py)
- [Embedding 评测脚本](../../scripts/evaluate_embedding_models.py)
- [反馈 A/B 脚本](../../scripts/feedback_ab_experiment.py)

真实正文、数字账号、群号、Provider 配置和数据库快照不得进入提交。每个可声称结果应保存数据集指纹、代码 commit、模型标识、提示协议版本、运行时间、随机种子、逐调用 usage、排除项和重试。

## 附录 E：当前可声称与不可声称

### 可声称

- Harrier 在这 47 题上的前 20 证据覆盖优于 MiniLM；相对 BM25 的 MRR 优势未被区间支持。
- B0 的 54 次线上观察全部只做预取，失败率 9.26% 且有明显时延长尾；语义质量未知。
- ECCR 已完成真实 Provider 两轮检索并在 #726 上形成证据闭合 brief；该例同步成本为 76,224 Token/379.394 秒。
- #726 的表层模型在 ECCR brief 下获得更高的原始模型 judge 分，但逐项证据复核排序为 B16 > ECCR > Control；这证明 surface reading 是独立瓶颈，不证明 ECCR 获胜。
- “好女孩”一次 ECCR 漏掉普通综合找到的核心语义；修复后的两轮审计找齐相关事实但没有构造独立语义桥，只能判部分成功。
- 账号身份在两个 cutoff 上可由宿主零调用精确绑定；LLM 不应拥有账号合并权。
- 四层路由、查询时论证图和持久边生命周期给出了可实现的下一版工程方案。

### 不可声称

- 0.16 的语义能力合格；
- ECCR、B16、full-MR 或缓存已经在总体上胜出；
- required-group recall 1.0 等于语义理解正确；
- “好女孩”测试已经通过；
- 表层 100/95 自动分等于人类准确率；
- 更多调用、更多图边或更长思考自然提高回答；
- 负面情绪本身证明旧记忆错误；
- 当前持久图已解决双时间、幂等、遗忘和所有别名歧义；
- 已完整复现 MRAgent 或达到其公开 benchmark。
