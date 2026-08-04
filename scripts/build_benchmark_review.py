from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .materialize_retrieval_benchmark import read_jsonl
except ImportError:  # Direct script execution.
    from materialize_retrieval_benchmark import read_jsonl


def _index(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(record[key]): record for record in records}


def build_payload(directory: Path, benchmark_path: Path | None = None) -> dict[str, Any]:
    corpus_path = directory / "corpus.jsonl"
    benchmark_path = benchmark_path or directory / "benchmark.jsonl"
    corpus = _index(read_jsonl(corpus_path), "doc_id")
    candidates = _index(read_jsonl(directory / "candidates.jsonl"), "candidate_id")
    benchmark = read_jsonl(benchmark_path)
    fingerprint = hashlib.sha256(
        benchmark_path.read_bytes() + b"\0" + corpus_path.read_bytes()
    ).hexdigest()[:20]

    items: list[dict[str, Any]] = []
    for item in benchmark:
        provenance = item.get("provenance") or {}
        candidate_id = str(provenance.get("candidate_id") or "")
        candidate = candidates[candidate_id]
        positive_ids = [str(value) for value in item["positive_doc_ids"]]

        context_ids: list[str] = []
        for field in ("target_context", "query_context"):
            for record in candidate.get(field) or []:
                doc_id = str(record.get("doc_id") or "")
                if doc_id in corpus and doc_id not in context_ids:
                    context_ids.append(doc_id)
        for doc_id in positive_ids:
            if doc_id not in context_ids:
                context_ids.append(doc_id)
        context_ids.sort(key=lambda doc_id: (int(corpus[doc_id]["sent_at"]), doc_id))

        decoy_ids = [
            str(value)
            for value in item.get("lexical_decoy_doc_ids") or []
            if str(value) in corpus and str(value) not in context_ids
        ]

        def document(doc_id: str) -> dict[str, Any]:
            source = corpus[doc_id]
            return {
                "doc_id": doc_id,
                "sent_at": int(source["sent_at"]),
                "speaker": str(source.get("speaker") or ""),
                "text": str(source.get("text") or ""),
                "model_positive": doc_id in positive_ids,
            }

        items.append(
            {
                "id": str(item["id"]),
                "candidate_id": candidate_id,
                "split": str(item["split"]),
                "confidence": float(item["confidence"]),
                "query": str(item["query"]),
                "query_time": int(item["query_time"]),
                "memory_type": str(item["memory_type"]),
                "epistemic": str(item["epistemic"]),
                "positive_doc_ids": positive_ids,
                "context": [document(doc_id) for doc_id in context_ids],
                "decoys": [document(doc_id) for doc_id in decoy_ids],
            }
        )
    return {
        "format_version": 1,
        "dataset_fingerprint": fingerprint,
        "items": items,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MR Memory 群聊测试集复核</title>
<style>
:root{color-scheme:light;--bg:#f5f7fb;--card:#fff;--ink:#1d2433;--muted:#687186;--line:#dce2ed;--blue:#2563eb;--green:#16835b;--red:#c73c46;--amber:#a16207;--soft:#eef3fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
.top{max-width:1180px;margin:auto;padding:16px 20px}.title-row,.toolbar,.meta,.decision-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
h1{font-size:22px;margin:0 auto 0 0}.privacy{color:#8a4b08;background:#fff7df;border:1px solid #f0d48a;border-radius:8px;padding:6px 10px}
.toolbar{margin-top:12px}.toolbar input,.toolbar select,.toolbar button,textarea,.field-select{font:inherit;border:1px solid #cbd3e1;border-radius:8px;background:#fff;color:var(--ink)}
.toolbar input{min-width:260px;padding:8px 10px}.toolbar select,.toolbar button{padding:8px 10px}.toolbar button{cursor:pointer}.primary{background:var(--blue)!important;color:#fff!important;border-color:var(--blue)!important}
.stats{font-variant-numeric:tabular-nums;color:var(--muted)}main{max-width:1180px;margin:20px auto;padding:0 20px 60px}.empty{text-align:center;padding:60px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;margin:0 0 16px;box-shadow:0 2px 10px rgba(35,52,80,.05);overflow:hidden}.card.accept{border-left:5px solid var(--green)}.card.reject{border-left:5px solid var(--red)}.card.edit{border-left:5px solid var(--amber)}
.card-head{padding:14px 16px;border-bottom:1px solid var(--line);background:#fbfcfe}.card-body{padding:16px}.meta{color:var(--muted)}.id{font-weight:750;color:var(--ink)}
.badge{padding:2px 7px;border-radius:999px;background:var(--soft);font-size:12px}.low{background:#fff1dc;color:#8a4b08}.human{font-weight:700}.human.accept{color:var(--green)}.human.reject{color:var(--red)}.human.edit{color:var(--amber)}
label.caption{display:block;font-weight:700;margin:0 0 6px}.query{width:100%;min-height:66px;padding:10px 12px;resize:vertical}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0}.field-select{width:100%;padding:8px 10px}
h2{font-size:15px;margin:18px 0 8px}.message{display:grid;grid-template-columns:24px 128px 1fr;gap:8px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin:6px 0;background:#fff}.message.model-positive{border-color:#8bb8ff;background:#f2f7ff}.message input{margin-top:4px}.who{font-weight:650}.time{display:block;color:var(--muted);font-size:11px}.doc{color:var(--muted);font-size:11px}.text{white-space:pre-wrap;word-break:break-word}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted)}.notes{width:100%;min-height:58px;padding:8px 10px;resize:vertical;margin-top:8px}
.decision-row{margin-top:14px}.decision-row button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;cursor:pointer}.decision-row button.active.accept{background:#e5f6ee;border-color:#72c6a5;color:#086642}.decision-row button.active.edit{background:#fff3da;border-color:#e1b966;color:#825300}.decision-row button.active.reject{background:#fdebed;border-color:#dfa0a6;color:#9d202d}.clear-one{margin-left:auto!important;color:var(--muted)}
@media(max-width:720px){.fields{grid-template-columns:1fr}.message{grid-template-columns:24px 1fr}.text{grid-column:2}.toolbar input{min-width:100%}.privacy{width:100%}}
</style>
</head>
<body>
<header><div class="top">
  <div class="title-row"><h1>MR Memory 群聊测试集复核</h1><span class="privacy">本页含私有群聊正文，请勿上传</span></div>
  <div class="toolbar">
    <input id="search" placeholder="搜索题号、问题或证据正文">
    <select id="filter"><option value="all">全部条目</option><option value="unreviewed">仅未复核</option><option value="accept">已接受</option><option value="edit">需修改</option><option value="reject">已拒绝</option><option value="ai_low_confidence">AI 低置信</option></select>
    <button class="primary" id="export">导出人类复核 JSON</button>
    <button id="importButton">导入复核 JSON</button><input id="importFile" type="file" accept="application/json" hidden>
    <span class="stats" id="stats"></span>
  </div>
</div></header>
<main id="cards"></main>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const dataset=JSON.parse(document.getElementById('dataset').textContent);
const storageKey='mr-memory-review:'+dataset.dataset_fingerprint;
const memoryTypes=[...new Set(dataset.items.map(x=>x.memory_type))].sort();
const epistemics=[...new Set(dataset.items.map(x=>x.epistemic))].sort();
let state={};
try{state=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(_){state={}}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=t=>new Date(Number(t)*1000).toLocaleString('zh-CN',{hour12:false});
const options=(values,current)=>values.map(v=>`<option value="${esc(v)}" ${v===current?'selected':''}>${esc(v)}</option>`).join('');
function initial(item){return {decision:'',query:item.query,memory_type:item.memory_type,epistemic:item.epistemic,positive_doc_ids:[...item.positive_doc_ids],notes:''}}
function itemState(item){return Object.assign(initial(item),state[item.id]||{})}
function save(){localStorage.setItem(storageKey,JSON.stringify(state));updateStats()}
function messageRow(doc,selected){return `<label class="message ${doc.model_positive?'model-positive':''}"><input class="evidence" type="checkbox" data-doc="${esc(doc.doc_id)}" ${selected?'checked':''}><span><span class="who">${esc(doc.speaker)}</span><span class="time">${fmt(doc.sent_at)}</span><span class="doc">${esc(doc.doc_id)}${doc.model_positive?' · AI 初标证据':''}</span></span><span class="text">${esc(doc.text)}</span></label>`}
function card(item){const s=itemState(item);const allDocs=[...item.context,...item.decoys];const selected=new Set(s.positive_doc_ids);const human=s.decision?`<span class="human ${s.decision}">人类复核：${{accept:'接受',edit:'需修改',reject:'拒绝'}[s.decision]}</span>`:'<span class="human">尚无人类复核</span>';return `<article class="card ${esc(s.decision)}" data-id="${esc(item.id)}">
<div class="card-head"><div class="meta"><span class="id">${esc(item.id)}</span><span class="badge ${item.split==='ai_low_confidence'?'low':''}">${esc(item.split)}</span><span class="badge">置信度 ${item.confidence.toFixed(2)}</span><span>${esc(item.candidate_id)}</span>${human}</div></div>
<div class="card-body"><label class="caption">检索问题（可直接修改）</label><textarea class="query">${esc(s.query)}</textarea>
<div class="fields"><label><span class="caption">记忆类型</span><select class="field-select memory-type">${options(memoryTypes,s.memory_type)}</select></label><label><span class="caption">认识状态</span><select class="field-select epistemic">${options(epistemics,s.epistemic)}</select></label></div>
<h2>候选上下文（勾选应作为正例的原消息）</h2>${item.context.map(d=>messageRow(d,selected.has(d.doc_id))).join('')}
<details><summary>自动词面干扰项 ${item.decoys.length} 条（未经负例确认）</summary>${item.decoys.map(d=>messageRow(d,selected.has(d.doc_id))).join('')||'<p>无</p>'}</details>
<label class="caption" style="margin-top:14px">复核备注</label><textarea class="notes" placeholder="错在哪里、应如何改，或为何接受">${esc(s.notes)}</textarea>
<div class="decision-row"><button data-decision="accept" class="${s.decision==='accept'?'active accept':''}">✓ 接受标签</button><button data-decision="edit" class="${s.decision==='edit'?'active edit':''}">✎ 需要修改</button><button data-decision="reject" class="${s.decision==='reject'?'active reject':''}">✕ 丢弃该题</button><button class="clear-one">清除此题复核</button></div></div></article>`}
function searchable(item){return [item.id,item.query,...item.context.map(x=>x.text),...item.decoys.map(x=>x.text)].join('\n').toLowerCase()}
function render(){const filter=document.getElementById('filter').value;const term=document.getElementById('search').value.trim().toLowerCase();const shown=dataset.items.filter(item=>{const s=itemState(item);if(term&&!searchable(item).includes(term))return false;if(filter==='all')return true;if(filter==='unreviewed')return !s.decision;if(filter==='ai_low_confidence')return item.split===filter;return s.decision===filter});document.getElementById('cards').innerHTML=shown.length?shown.map(card).join(''):'<div class="empty">没有符合条件的条目</div>';updateStats()}
function capture(cardEl){const id=cardEl.dataset.id;const item=dataset.items.find(x=>x.id===id);const s=itemState(item);s.query=cardEl.querySelector('.query').value.trim();s.memory_type=cardEl.querySelector('.memory-type').value;s.epistemic=cardEl.querySelector('.epistemic').value;s.notes=cardEl.querySelector('.notes').value.trim();s.positive_doc_ids=[...cardEl.querySelectorAll('.evidence:checked')].map(x=>x.dataset.doc);state[id]=s;save();return s}
document.getElementById('cards').addEventListener('change',e=>{const c=e.target.closest('.card');if(c)capture(c)});
document.getElementById('cards').addEventListener('input',e=>{const c=e.target.closest('.card');if(c)capture(c)});
document.getElementById('cards').addEventListener('click',e=>{const c=e.target.closest('.card');if(!c)return;const id=c.dataset.id;if(e.target.dataset.decision){const s=capture(c);s.decision=e.target.dataset.decision;state[id]=s;save();render()}else if(e.target.classList.contains('clear-one')){delete state[id];save();render()}});
document.getElementById('filter').addEventListener('change',render);document.getElementById('search').addEventListener('input',render);
function updateStats(){const values=dataset.items.map(item=>itemState(item));const reviewed=values.filter(x=>x.decision).length;const accepted=values.filter(x=>x.decision==='accept').length;const edited=values.filter(x=>x.decision==='edit').length;const rejected=values.filter(x=>x.decision==='reject').length;document.getElementById('stats').textContent=`已复核 ${reviewed}/${values.length} · 接受 ${accepted} · 修改 ${edited} · 丢弃 ${rejected}`}
document.getElementById('export').addEventListener('click',()=>{document.querySelectorAll('.card').forEach(c=>capture(c));const payload={format_version:1,dataset_fingerprint:dataset.dataset_fingerprint,reviewer_type:'human',exported_at:new Date().toISOString(),reviews:dataset.items.map(item=>({id:item.id,...itemState(item)}))};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`human-review-${dataset.dataset_fingerprint}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
document.getElementById('importButton').addEventListener('click',()=>document.getElementById('importFile').click());
document.getElementById('importFile').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{const payload=JSON.parse(await file.text());if(payload.dataset_fingerprint!==dataset.dataset_fingerprint)throw new Error('测试集指纹不一致');state={};for(const review of payload.reviews||[]){if(dataset.items.some(x=>x.id===review.id))state[review.id]=review}save();render()}catch(err){alert('导入失败：'+err.message)}finally{e.target.value=''}});
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private interactive human-review page.")
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        help="Benchmark JSONL to review; defaults to benchmark.jsonl in benchmark_dir.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build_payload(args.benchmark_dir, args.benchmark_file)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    output = args.output or args.benchmark_dir / "review.html"
    output.write_text(HTML_TEMPLATE.replace("__DATA__", encoded), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "items": len(payload["items"]),
                "dataset_fingerprint": payload["dataset_fingerprint"],
                "bytes": output.stat().st_size,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
