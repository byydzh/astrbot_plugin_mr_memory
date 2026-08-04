# MR Memory

面向 AstrBot 群聊的证据可追溯记忆插件脚手架。目标是逐步替代
AngelEye 的历史检索和 Local Reminiscence 的语义记忆。当前版本提供原始消息
真值层、论文同构的图结构、LLM 主动遍历工具接口和离线回放能力。

研究基础：[Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036)。

> [!WARNING]
> 本插件把群聊隔离视为独立安全边界，不信任主 Agent 替它完成隔离。
> 作用域只从当前群事件的 UMO 推导，每个群使用独立 SQLite 文件，且 SQL
> 仍会校验 scope；LLM 和用户参数都不能指定其他群。
> 详细 invariant 与威胁边界见 [架构文档](docs/ARCHITECTURE.md)。

## 安全默认值

- `capture_enabled=false`：不采集任何线上消息。
- `subconscious_provider_id=deepseek/deepseek-v4-flash`：记忆推理与主 LLM 分离。
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

插件通过 AstrBot 的 `Context.tool_loop_agent()` 调用自己配置的 provider，不继承
当前会话主 LLM。若当前会话还没有图记忆，自动唤醒会直接跳过，不产生模型调用。

## 离线回放

在本插件仓库根目录执行：

```powershell
python -m scripts.replay_fixture `
  --input dev/fixtures/sample_messages.jsonl `
  --database .dev/mr-memory/replay.db
```

回放是幂等的，同一 `source_key` 重复导入不会生成重复消息。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 当前边界

- 尚未实现 LLM 记忆抽取，因此图层需要由后续蒸馏管线填充。
- 尚未下载或分析图片，仅允许在 `content` 中保留附件元数据。
- 私有 LLM 已可主动组合遍历工具；cue 初始化、episode 分段和蒸馏仍待实现。
- 定时/定量后台整理尚未实现，当前只有主 LLM 请求前唤醒和按需咨询。
- 线上部署、历史迁移和双轨对比不属于 0.3.0 脚手架范围。
