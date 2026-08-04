# MR Memory

面向 AstrBot 群聊的证据可追溯记忆插件原型。目标是逐步替代
AngelEye 的历史检索和 Local Reminiscence 的语义记忆。当前版本提供原始消息
真值层、LLM 图构建、embedding 候选初始化、主动遍历和离线回放能力。

研究基础：[Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036)。

> [!WARNING]
> 本插件把群聊隔离视为独立安全边界，不信任主 Agent 替它完成隔离。
> 作用域只从当前群事件的 UMO 推导，每个群使用独立 SQLite 文件，且 SQL
> 仍会校验 scope；LLM 和用户参数都不能指定其他群。
> 详细 invariant 与威胁边界见 [架构文档](docs/ARCHITECTURE.md)。

## 安全默认值

- `capture_enabled=false`：不采集任何线上消息。
- `subconscious_provider_id=deepseek/deepseek-v4-flash`：记忆推理与主 LLM 分离。
- `embedding_model_name=BAAI/bge-small-zh-v1.5`：插件本地运行的中文 ONNX
  embedding 模型，不经过 AstrBot Embedding Provider 或远程推理 API。
- `expose_traversal_tools=false`：主 LLM 默认看不到七个底层遍历工具。
- `consult_tool_enabled=true`：主 LLM 只看到一个潜意识咨询桥接工具。
- 数据库固定写入本插件的 `plugin_data` 目录。
- 日志默认不输出聊天正文。
- 不连接或重启 NapCat，不发送消息。

开发隔离由“只在本地/克隆环境运行、不部署线上”保证，不通过阉割 MRAgent 的
LLM 工具接口实现。

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
        +-- Person--Aspect--Semantic
        +-- Topic--Episode
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
        +-- one optional consultation tool visible to the main LLM
```

`mr_memory/` 不依赖 AstrBot，可以通过普通 Python 测试和 JSONL 回放独立运行。
`main.py` 负责把 AstrBot 事件转换为核心模型。论文 Table 4 的七类工具只加入
插件私有 LLM 的工具循环；主 LLM 默认只能接收有长度上限的证据摘要，或调用
`mr_consult_subconscious` 请求一次更聚焦的重建。

插件使用与 AstrBot `Context.tool_loop_agent()` 相同的 `ToolLoopAgentRunner` 调用自己
配置的 provider，不继承当前会话主 LLM。插件直接持有 runner，以便记录整个多轮循环的
聚合 token，而不是误把最终 response 当成全部费用。若当前会话还没有图记忆，自动唤醒
会直接跳过，不产生模型调用。

管理员可在目标群执行 `/mrmem distill`，使用潜意识 Provider 将最近一批原始消息
构造成 episode、cue/tag、语义记忆和 topic；启用本地 embedding 后会同时建立
同群向量索引。LLM 返回的 source key 会在落库前校验，不能引用批次外消息。

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

- 所有已知群范围的消息量、图记忆量、向量量和数据库占用概览；
- Cue--Tag--Episode、Person--Aspect--Semantic、Topic--Episode 图谱；
- 点击 Episode 回溯关键词和原始聊天证据；
- 按正文或发送者检索当前群的原始消息；
- 明确提示模型费用后，手动触发当前群的最近消息整理。

控制台只使用服务端枚举的 64 位不透明 scope ID。后端会再次从对应数据库读取并
校验 UMO、平台和群号，不接受前端直接指定任意 UMO。

管理员可在群内使用 `/mrmem usage [limit]` 查看该群最近的构建与重建 token ledger。
记录按群隔离，不保存隐藏推理或完整模型输出。

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

## 测试

```powershell
python -m unittest discover -s tests -v
```

真实群聊 embedding 选型目前使用证据式模型初标集，构建规则、隐私边界和评测口径见
[群聊检索测试集](docs/RETRIEVAL_BENCHMARK.md)。真实正文只保存在 Git 忽略的
`.dev/`，不会进入仓库或外部推理服务。

## 当前边界

- 尚未下载或分析图片，仅允许在 `content` 中保留附件元数据。
- 图构建目前由管理员显式触发；定时/定量后台整理尚未接入。
- 宿主直接证据门控已在离线消融中验证，尚未接入运行时 runner。
- semantic memory 聚合多条消息时仍需补齐多源 provenance 和反馈修订状态机。
- 当前开发版本不自动部署线上；真实历史的首个遮罩 A/B 已完成，线上 canary 尚未开启。
