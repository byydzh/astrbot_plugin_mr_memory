# 群聊检索测试集

这个测试集用于比较本地 embedding 和后续图记忆检索策略，不把聊天中的一句话
直接当作现实事实。标注目标是“哪几条消息构成回答所需的原始证据”，并另外记录
它属于陈述、转述、不确定说法、主观意见、玩笑或后续纠正。

## 构建方法

1. 只读打开 Angel Eye 的群聊历史 SQLite。
2. 每个群独立处理；群号、QQ 号和昵称不进入导出文件。
3. URL、邮箱、IP、长数字标识和疑似密钥在导出时替换为占位符。
4. 用间隔至少一小时、且能回指原消息的回复链生成候选，避免凭空猜正例。
5. 由模型直接把候选改写为可独立理解的检索问题，并逐条指定证据消息。
6. 指代或图片上下文不足的条目标成 `ai_low_confidence`，其余标成
   `ai_high_confidence`。两者都尚未经过人类复核，不能称为 gold。

时间评测只允许检索 `document.sent_at < query.query_time` 的消息；执行器还必须先按
`scope_id` 过滤，禁止跨群候选。`positive_doc_ids` 是模型第一轮直接标出的证据，
`lexical_decoy_doc_ids` 只是自动生成的压力样本。两者都不是经人类确认的标签。

## 私有输出

真实语料默认放在仓库根目录的：

```text
.dev/benchmarks/group_retrieval_v1/
  corpus.jsonl       脱敏后的单群文本语料
  candidates.jsonl   回复链候选、上下文和词面干扰项
  annotations.jsonl  模型第一轮直接标注
  benchmark.jsonl    校验后可评测数据
  manifest.json      数量、隐私声明和评测约束
```

`.dev/` 已被 Git 忽略。这些文件仍含群聊正文，不能提交或发送给外部推理 API。

## 校验和词面基线

在插件目录的上一级执行：

```powershell
python astrbot_plugin_mr_memory/scripts/materialize_retrieval_benchmark.py `
  .dev/benchmarks/group_retrieval_v1

python astrbot_plugin_mr_memory/scripts/evaluate_retrieval_benchmark.py `
  .dev/benchmarks/group_retrieval_v1 --split ai_high_confidence
```

生成供人类逐题检查的本地交互页：

```powershell
python astrbot_plugin_mr_memory/scripts/build_benchmark_review.py `
  .dev/benchmarks/group_retrieval_v1
```

打开生成的 `review.html` 后，可以修改问题和标签、增删正例证据、选择接受/修改/
丢弃并填写备注。状态只保存在浏览器本地；“导出人类复核 JSON”会生成带数据集
指纹的复核结果，原测试集文件不会被页面直接改写。

导出后，将人类决定应用到测试集：

```powershell
python astrbot_plugin_mr_memory/scripts/apply_human_review.py `
  .dev/benchmarks/group_retrieval_v1 `
  "C:/Users/<用户名>/Downloads/human-review-<fingerprint>.json"
```

脚本会校验数据集指纹、群作用域、证据存在性和时间边界，生成：

- `benchmark_gold.jsonl`：仅包含人类明确接受的条目；
- `benchmark_human_reviewed.jsonl`：同时保留接受、退回、拒绝和未复核状态；
- `human_review.json` 与 `human_review_summary.json`：原始复核记录和统计。

如果退回修改的条目已经在交互中被人类逐条或整体明确接受，可直接记录该次批准，
无需再次导出浏览器文件：

```powershell
python astrbot_plugin_mr_memory/scripts/finalize_gold_benchmark.py `
  .dev/benchmarks/group_retrieval_v1 `
  --approval-source explicit_user_approval_in_codex_task
```

这会把第一轮 gold 与获批修订合并为 `benchmark_gold_final.jsonl`，并保留两轮
批准来源。随后可在完整语料、严格群作用域和时间过滤下运行本地 embedding 矩阵：

```powershell
python astrbot_plugin_mr_memory/scripts/evaluate_embedding_models.py `
  .dev/benchmarks/group_retrieval_v1 --device cuda
```

人类新增了较晚证据时，查询时间会移动到该证据之后；若人类移除了原先作为
查询来源的回复消息，查询时间会回退到剩余证据之后。两种变化都会写入 provenance，
防止悄悄产生未来消息泄漏。

词面基线使用中文字符和二元组 BM25，只用于检查数据是否可检索。当前只能在同一
`ai_high_confidence` 初标集上比较 MRR、Hit Rate、平均证据 Recall 和完整证据
Recall；经过人类逐条复核后才能冻结为 gold 测试集。
