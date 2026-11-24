import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from mr_memory.store import Store
from mr_memory import manual_messages as manual
from mr_memory.network import writer, recall
from mr_memory.network.storage import SCHEMA
from mr_memory.network.storage import opened
from mr_memory.network.projection import full


def test_catchup_bridge_uses_witness_dates_and_releases_retired_support(tmp_path):
    store = Store(tmp_path / "bridge.db", "test:GroupMessage:bridge")
    db = store.db
    db.executescript(SCHEMA)
    keys = ["M|episode:1", "M|episode:2"]
    with db:
        for k, stamp in zip(keys, [1704153600, 1712102400]):
            db.execute("INSERT INTO node VALUES(?,'M','episode','既有经历',?,'[]','ACTIVE',1)", (k, stamp))
            db.execute("INSERT INTO network_l1_clock VALUES(?,?)", (k, stamp))
    net = opened(db)
    op = dict(id="+b", region="M", kind="bridge", text="两条经历有明确联系", subjects=[], basis=keys)
    result = net.apply("catchup", 1720000000, [], keys, dict(changes=[op], edges=[], incoming_l1=[keys[1]]))
    bridge = result["edits"][0]["id"]
    assert tuple(db.execute("SELECT via,src,dst FROM derived_shortcut").fetchone()) == (
        bridge,
        keys[1],
        keys[0],
    )
    assert db.execute("SELECT active FROM edge WHERE rel='bridge'").fetchone()[0] == 1
    net.apply(
        "retire",
        1720001000,
        [],
        [bridge, *keys],
        dict(changes=[dict(op, id=bridge, revision=1, status="RETRACTED")], edges=[], incoming_l1=[]),
    )
    assert not db.execute("SELECT 1 FROM derived_shortcut").fetchone()
    assert not db.execute("SELECT 1 FROM edge WHERE rel='bridge'").fetchone()
    store.close()


def test_migrated_scope_consumes_once_and_reopens_with_evidence(tmp_path):
    async def exercise():
        umo = "test:GroupMessage:one"
        store = Store(tmp_path / "scope.db", umo)
        store.db.executescript(SCHEMA)
        with store.db:
            store.db.execute("INSERT INTO network_scope VALUES(?,?,0)", (umo, "fixture"))
        store.close()
        store = Store(tmp_path / "scope.db", umo)
        raw = store.append_message(
            dict(
                platform_id="test",
                group_id="one",
                sender_id="alice",
                sender_name="小林",
                sent_at=11,
                message_id="a",
                plain_text="我平时叫时缎。",
            )
        )
        bad = dict(
            request_id="bad", messages=[{"text": "第一条"}, {"text": "第二条", "reply_to_message_id": 99}]
        )
        with pytest.raises(ValueError):
            manual.submit(store, bad)
        assert store.db.execute("SELECT count(*) FROM messages").fetchone()[0] == 1
        body = dict(
            request_id="note",
            trigger="now",
            messages=[
                {"text": "补录原话", "mentioned_account_ids": ["alice"], "reply_to_message_id": raw["id"]}
            ],
        )
        assert manual.submit(store, body) == ("note", True)
        assert manual.submit(store, body) == ("note", False)
        with pytest.raises(ValueError):
            manual.submit(store, dict(body, trigger="next_window"))
        vec = np.zeros(640, dtype="<f4")
        vec[0] = 1
        emb = SimpleNamespace(
            model_id="fixture",
            query=AsyncMock(return_value=vec.tolist()),
            texts=AsyncMock(side_effect=lambda t: [vec.tolist() for _ in t]),
        )
        answers = [
            {"items": [["episode", ["u0001"], "群友说明自己的称呼", [1, 2]]]},
            {
                "changes": [
                    {
                        "id": "+a",
                        "region": "S",
                        "kind": "understanding",
                        "text": "小林曾说明称谓",
                        "subjects": ["u0001"],
                        "basis": ["m1", "r1"],
                    }
                ],
                "edges": [],
            },
        ]
        agent = SimpleNamespace(
            max_output_tokens=4000,
            timeout_seconds=3,
            _model_turn=AsyncMock(
                side_effect=[
                    SimpleNamespace(completion_text=json.dumps(v), usage={"output": 33}) for v in answers
                ]
            ),
        )
        cfg = dict(
            embedding_enabled=True,
            distillation_max_messages=10,
            private_daily_token_budget=100000,
            learning_window_enabled=False,
            auto_distillation_min_pending=2,
            maintenance_interval_seconds=100,
        )
        plugin = SimpleNamespace(agent=lambda *args, **kw: agent, embedder=emb, config=cfg)
        first = await writer.consolidate(plugin, store, True)
        assert first["written_count"] == 1 and manual.receipt(store, "note")["state"] == "running"
        store.close()
        store = Store(tmp_path / "scope.db", umo)
        second = await writer.consolidate(plugin, store, True)
        assert second["written_count"] == 1
        receipt = manual.receipt(store, "note")
        assert receipt["state"] == "completed" and len(receipt["runs"]) == 2
        assert receipt["runs"][1]["changes"][0]["after"] == "小林曾说明称谓"
        assert any(e["active_after"] == 1 for e in receipt["runs"][1]["edges"])
        assert (await writer.consolidate(plugin, store, True))["status"] == "idle"
        assert full(store.db)[1] == 0
        entrances = recall.locate(store, "fixture", [vec], dict(sender_id="alice", plain_text="我来了"), [])
        assert entrances["人物候选"] == ["u0001"]
        assert any(v["region"] == "S" for v in entrances["旧图入口"])
        store.forget("alice")
        assert not any(
            v["region"] == "S"
            for v in recall.locate(store, "fixture", [vec], dict(plain_text="我来了"), [])["旧图入口"]
        )
        store.close()

    asyncio.run(exercise())
