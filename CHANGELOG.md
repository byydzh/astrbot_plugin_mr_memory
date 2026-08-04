# Changelog

## 0.7.0

- Add strict historical cutoffs to graph search and all seven traversal tools.
- Add reverse-window masked-call construction and controlled A/B tooling for real
  retained group calls without future leakage.
- Add privacy-minimized experiment, token-usage, and reconstruction-step ledgers.
- Record full private-agent aggregate usage at runtime instead of counting only
  the final response, and expose recent per-group totals through `/mrmem usage`.
- Validate a deterministic host evidence gate that stops needless graph browsing
  once a high-score episode is verified against raw source messages.

## 0.6.0

- Add a plugin-owned Sentence Transformers backend alongside FastEmbed/ONNX.
- Preserve checkpoint dtype with `dtype=auto` and support query-only named
  prompts required by asymmetric models such as Harrier.
- Bound Sentence Transformers batch size and maximum sequence length for
  low-memory deployments.
- Add an opt-in startup preload probe for deployment resource verification.

## 0.5.0

- Add an AstrBot-native Plugin Page console under `pages/console`.
- Add authenticated Web APIs for scope overview, graph inspection, source-message
  search, episode evidence, and explicit distillation.
- Visualize Cue--Tag--Episode, Person--Aspect--Semantic, and Topic--Episode links.
- Persist and verify each physical SQLite database's group-scope identity.
- Keep console scope selection opaque and server-resolved; raw UMO values are never
  accepted as API routing inputs.

## 0.4.0

- Reproduce the paper's validated episode, cue/tag, semantic-memory, and topic
  construction path.
- Add a plugin-owned local FastEmbed/ONNX backend and a dependency-free hash
  backend for offline tests; no AstrBot Embedding Provider or remote embedding
  API is used.
- Initialize reconstruction from vector-matched cues, episodes, and topics.
- Add `/mrmem distill` for explicit per-group graph construction.
- Inject dynamic memory evidence as temporary user content on AstrBot 4.27.
- Add an end-to-end offline reproduction fixture and regression test.

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
