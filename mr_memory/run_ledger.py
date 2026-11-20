"""Lossless private run records. Keep tool results in their committed step."""
import base64
import json
import zlib
from collections import defaultdict


STORAGE_KEY = "_mr_run_storage_v2"


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def pack_run(payload, steps):
    """Only replace bodies for which this run already has a matching durable step."""
    results = defaultdict(list)
    for step in steps:
        data = step["data"]
        if data.get("tool_call_id") and "result" in data:
            results[data["tool_call_id"]].append((step["seq"], data.get("name"), _wire(data["result"])))

    packed = dict(payload)
    result_links = []
    text_links = []
    for field in ("tool_calls", "messages"):
        records = payload.get(field)
        if not isinstance(records, list):
            continue
        rows = list(records)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if field == "tool_calls" and "result" in row:
                candidates = results.get(row.get("id"), ())
                if not candidates:
                    continue
                encoded = _wire(row["result"])
                for seq, name, body in candidates:
                    if name == row.get("name") and encoded == body:
                        rows[index] = {k: v for k, v in row.items() if k != "result"}
                        result_links.append([index, seq])
                        break
            elif field == "messages" and row.get("role") == "tool" and isinstance(row.get("content"), str):
                candidates = results.get(row.get("tool_call_id"), ())
                if len(candidates) != 1:
                    continue
                seq, _, body = candidates[0]
                text = row["content"].encode("utf-8")
                link = [index, seq]
                if text != body:
                    # The model often saw a projection, with omitted fields or prior-message
                    # references. Store its exact bytes relative to the original result.
                    if len(text) < 256:
                        continue
                    encoder = zlib.compressobj(zdict=body[-32768:])
                    delta = base64.b64encode(encoder.compress(text) + encoder.flush()).decode("ascii")
                    if len(delta) + 32 >= len(text):
                        continue
                    link.append(delta)
                rows[index] = {k: v for k, v in row.items() if k != "content"}
                text_links.append(link)
        packed[field] = rows
    if result_links or text_links:
        packed[STORAGE_KEY] = {"results": result_links, "texts": text_links}
    return packed


def unpack_run(payload, steps):
    """Hydrate read views only. Older rows have inline bodies and need no conversion."""
    links = payload.pop(STORAGE_KEY, None)
    if links is None:
        return payload
    by_seq = {s["seq"]: s["data"] for s in steps}
    for index, seq in links["results"]:
        call = payload["tool_calls"][index]
        data = by_seq[seq]
        if (call["id"], call["name"]) != (data["tool_call_id"], data["name"]):
            raise ValueError("Stored tool result refers to a different call")
        call["result"] = data["result"]
    for index, seq, *delta in links["texts"]:
        message = payload["messages"][index]
        data = by_seq[seq]
        if message["tool_call_id"] != data["tool_call_id"]:
            raise ValueError("Stored transcript refers to a different call")
        body = _wire(data["result"])
        if delta:
            decoder = zlib.decompressobj(zdict=body[-32768:])
            body = decoder.decompress(base64.b64decode(delta[0], validate=True)) + decoder.flush()
            if not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid stored transcript")
        message["content"] = body.decode("utf-8")
    return payload
