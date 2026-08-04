# Changelog

## 0.3.0

- Add a plugin-owned provider, defaulting to
  `deepseek/deepseek-v4-flash`, for memory reconstruction.
- Run the seven graph traversal tools inside a bounded private tool loop.
- Hide low-level traversal tools from the main LLM by default.
- Add automatic pre-request memory briefs and one optional high-level
  `mr_consult_subconscious` bridge tool.
- Skip the provider call entirely when the current session has no graph units.
- Enforce group isolation with event-derived scopes, one SQLite file per group,
  scoped SQL joins, and cross-group regression tests.

## 0.2.0

- Replace the passive all-in-one LLM search tool with the seven typed memory
  traversal tools from MRAgent Table 4.
- Add Cue--Tag--Episode, Person--Aspect--Semantic, and Topic--Episode schema.
- Enable the traversal toolkit by default when the plugin is explicitly loaded;
  development isolation remains a deployment boundary.

## 0.1.0

- 建立默认关闭的 AstrBot 事件适配层。
- 建立 SQLite 真值层、FTS5 索引和图记忆预留表。
- 增加幂等 JSONL 离线回放工具。
- 增加群隔离、全文检索和回放测试。
