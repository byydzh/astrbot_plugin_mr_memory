# Changelog

## 0.12.1

- Initialize tool visibility, maintenance workers, provider checks, and saved
  distillation queues on both cold startup and zero-restart plugin reload.
- Discover existing per-group databases after reload so pending historical
  construction resumes without waiting for a new message in each group.
- Distinguish the full OneBot group inventory from AngelEye's on-demand cache;
  the console now labels the number of materialized MR databases separately
  from the actual joined-group count and reports how many groups have no cached
  history source.
- Give background construction and feedback maintenance their own 300-second
  timeout while keeping the interactive reconstruction timeout at 45 seconds.
- Keep DeepSeek V4 construction thinking enabled by default and consume its stream
  so long reasoning calls are not mistaken for a dead non-streaming request. The
  setting remains explicit for diagnostics instead of silently downgrading the model.
- Bound construction to 40 source messages and a non-constraining 32768 output-token
  safety ceiling; concise graph-unit counts and the per-group daily ledger control
  cost while normal calls stop naturally.
- Work around AstrBot 4.27.1's dropped OpenAI-provider generation kwargs inside
  the plugin, without patching AstrBot core or changing the selected Provider.
- Retry terminal maintenance jobs only on explicit runtime bootstrap, so a
  fixed deployment recovers old failures without creating an infinite sweep loop.
- Requeue messages that exhausted their three construction attempts only at that
  same explicit reload boundary, including small groups with no ordinary pending
  messages left.
- Log only response lengths, token counts, and elapsed time for construction so
  provider-mode failures are diagnosable without exposing group-chat content.
- Retry deterministic JSON/evidence validation once with a bounded repair request;
  strict host validation remains unchanged and repeated blind maintenance retries
  are avoided.
- Deterministically discard isolated invalid optional graph units, re-run the same
  strict validator, and audit any newly uncovered raw evidence instead of failing
  an otherwise useful 40-message batch.
- Verify the exact 40-message production prompt with thinking enabled: the model
  stopped naturally after 22,094 completion tokens and about 200 seconds; the host
  then rejected one unsafe participant binding without weakening identity rules.

## 0.12.0

- Make the local embedding backend and model first-class settings instead of
  hiding the selected model behind a boolean switch.
- Expose the actual runtime triggers: automatic wake on every main-LLM request
  versus tool-only consultation, message-count construction threshold, maximum
  construction delay, feedback attribution window, and the high-level consult
  bridge.
- Correct UMO guidance to use the platform *instance* ID rather than the adapter
  type. An empty scope list explicitly means all groups.
- Add a read-only AngelEye history detector and a background, idempotent,
  group-isolated import flow. The console shows source coverage, target counts,
  platform mapping, progress, malformed-row skips, and the LLM cost boundary
  before starting.
- Show the effective scope, wake, construction, feedback, and embedding policies
  in the console instead of requiring users to infer them from hidden defaults.

## 0.11.0

- Persist explicit `HYPOTHESIS`, `SUPPORTED`, `CONTESTED`, and `CONFIRMED`
  states plus an uncertainty note on learned association edges. A repeated upsert
  cannot silently overwrite the epistemic state; later evidence must use the
  auditable `revise_edge` operation.
- Let ordinary incremental construction create new group-local association
  hypotheses and revise existing ones, so semantic learning is not limited to
  feedback on a Bot response. Competing interpretations remain separate graph paths.
- Add a bounded per-group repeated-media index that stores only adapter reference
  hashes, counts, sender cardinality, and a tiny source-key reservoir. No media bytes,
  URLs, OCR, captions, or automatic vision calls are retained.
- Give the private agent a read-only repeated-media context tool and explicitly ban
  visual inference from opaque hashes. The console renders unresolved association
  paths as dashed lines and exposes their uncertainty in the inspector.
- Reduce the normal settings surface to seven operational choices. Low-level tuning
  remains schema-backed for compatibility but is hidden from the dashboard.

## 0.10.0

- Add a group-scoped plastic association graph for learned meanings, symbols,
  behaviors, preferences, procedures, and traversal paths while keeping account
  identity and raw provenance in the deterministic truth layer.
- Let the private LLM register and version relation types, and propose evidence-bound
  edge upserts, reinforcement, inhibition, retirement, relation revision, and
  reversible node merges under host validation.
- Treat local embedding distance only as a candidate-generation prior. Run a bounded
  semantic tick for every main-LLM request once the group has graph memory; the
  private LLM owns relevance and stopping, with the deterministic early-stop gate
  retained only as an optional optimization.
- Persist bounded subconscious operational state and maintenance jobs so restarts do
  not erase focus, active edge IDs, or queued work; hidden reasoning is never stored.
- Record plastic-edge activation in observable interaction traces and assign feedback
  credit only to paths that actually influenced the eligible response. A graph
  mutation is rejected until its feedback proposal is host-committed.
- Extend the authenticated console, dashboard metrics, runtime ledger, and regression
  suite for dynamic relation and plastic graph inspection.
- Add a real-history local-semantics A/B harness with self-contained SVG reports and
  complete per-call token accounting for the “好女孩” and “阿拉蕾” cases.

## 0.9.0

- Add truth-layer Participant identities keyed by group scope, platform, and account
  ID, with time-bounded alias history, exact ambiguity handling, structured mention
  and reply relations, and administrator-confirmed aliases without account merging.
- Capture visible Bot output as ordinary `BOT` evidence linked by `RESPONDS_TO`, and
  handle platform recalls, edits, revisions, derived-memory invalidation, and
  self-service account erasure with future-capture suppression.
- Replace recent-window construction with per-message checkpoints, oldest-first
  batches, overlap context, retry bounds, content-hash verification, stable
  evidence-set episode keys, interrupted-batch recovery, and an explicit
  cited-or-ignored coverage ledger.
- Replace free-form person memory with structured claims that separate speaker and
  subject, validate exact evidence spans, support multiple sources, mark epistemic
  state, quarantine high-risk or uncertain claims, and implement conflict,
  supersede, retract, stale, and revision states. Quarantined claims remain outside
  automatic retrieval until independent-source promotion.
- Compute authoritative episode time from source messages and rebuild topic summaries
  from their currently linked active episodes.
- Add owner-type vector quotas, a minimum similarity threshold, a cheap history-intent
  gate, and the validated host evidence stop gate to the runtime reconstruction loop.
- Require the private reconstruction agent to return machine-validated
  claim/source/conflict/unresolved JSON; every item cites visited evidence, and
  truncation occurs only at complete structural units.
- Move automatic construction and feedback maintenance to a bounded background queue;
  add a lexical/reply feedback shutter and a per-group rolling private-token budget.
- Split optional Sentence Transformers/Harrier dependencies from the default
  FastEmbed install.
- Extend the authenticated console with Participant nodes, identity metrics, alias
  history, ambiguity reporting, pending checkpoints, and administrator alias binding.
- Add AstrBot 4.27.1 runtime-contract CI and truth-v2 regression coverage.

## 0.8.0

- Add a persistent, inspectable interaction graph for main-agent requests, tool
  actions, visible responses, later feedback, and prospective hypotheses without
  storing hidden chain-of-thought.
- Add a private feedback-maintenance agent with evidence inspection, scoped
  hypothesis lookup, and one atomic host-validated mutation transaction.
- Add signed backward utility assignment while keeping factual evidence confidence
  independent from behavioral feedback.
- Add explicit `always` and `semantic` activation modes, a lexical fast gate, and
  private-agent semantic activation for paraphrases.
- Add configurable commit thresholds, strict group/sender/cutoff checks, bounded
  media-redacted evidence, proposal binding, and fail-open runtime hooks.
- Add decay, dormancy, bounded active views, and reversible hypothesis merges
  without deleting provenance.
- Extend the AstrBot Plugin Page graph with Action--Feedback--Prospective Hypothesis
  nodes and add a strict real-history feedback A/B harness with token accounting.

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
