from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


def _bounded_text(
    value: object,
    field: str,
    *,
    limit: int,
    required: bool = True,
) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{field} is required")
    if len(text) > limit:
        raise ValueError(f"{field} exceeds {limit} characters")
    return text


def _revision_token(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer or string revision token")
    token = str(value).strip()
    if not token or len(token) > 160:
        raise ValueError(f"{field} is not a bounded revision token")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{field} must be non-negative")
    return token


def _sha256(value: object, field: str) -> str:
    digest = str(value or "").strip().casefold()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_certificate_lookup_key(
    snapshot: "RequestSnapshot",
    *,
    packet_sha256: str,
) -> str:
    """Return the stable semantic-cache/singleflight key for one packet.

    Request-local ids, cutoffs and data heads are deliberately absent.  A caller
    may reuse the associated certificate only after rebuilding the exact packet,
    re-auditing every cited source against the current snapshot, and rebinding the
    certificate envelope to that snapshot.
    """

    packet_hash = _sha256(packet_sha256, "packet_sha256")
    return stable_sha256(
        {
            "scope": snapshot.scope_sha256,
            "query": snapshot.query_sha256,
            "context": snapshot.context_sha256,
            "sender": snapshot.sender_participant_key,
            "reply": snapshot.reply_source_key,
            "packet": packet_hash,
            "reader_model": snapshot.inference_revision.reader_model,
            "reader_protocol": snapshot.inference_revision.reader_protocol,
            "certificate_schema": snapshot.inference_revision.certificate_schema,
            "surface_compiler": snapshot.inference_revision.surface_compiler,
            "route_policy": snapshot.inference_revision.route_policy,
        }
    )


@dataclass(frozen=True, slots=True)
class DataRevisionVector:
    """Host-owned heads for data that can change memory visibility or meaning."""

    message: str
    deletion: str
    identity: str
    graph: str
    relation: str
    feedback: str

    _FIELDS = ("message", "deletion", "identity", "graph", "relation", "feedback")

    @classmethod
    def from_value(cls, value: object) -> DataRevisionVector:
        if not isinstance(value, Mapping):
            raise ValueError("data_revision must be an object")
        keys = {str(key) for key in value}
        if keys != set(cls._FIELDS):
            missing = sorted(set(cls._FIELDS) - keys)
            unknown = sorted(keys - set(cls._FIELDS))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("data_revision fields are invalid: " + "; ".join(details))
        return cls(
            **{
                field: _revision_token(value[field], f"data_revision.{field}")
                for field in cls._FIELDS
            }
        )

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self._FIELDS}

    @property
    def digest(self) -> str:
        return stable_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class InferenceRevisionVector:
    """Revisions that affect retrieval, semantic reading, routing, or rendering."""

    retriever: str
    embedding_model: str
    fusion_policy: str
    reader_model: str
    reader_protocol: str
    certificate_schema: str
    surface_compiler: str
    route_policy: str

    _FIELDS = (
        "retriever",
        "embedding_model",
        "fusion_policy",
        "reader_model",
        "reader_protocol",
        "certificate_schema",
        "surface_compiler",
        "route_policy",
    )

    @classmethod
    def from_value(cls, value: object) -> InferenceRevisionVector:
        if not isinstance(value, Mapping):
            raise ValueError("inference_revision must be an object")
        keys = {str(key) for key in value}
        if keys != set(cls._FIELDS):
            missing = sorted(set(cls._FIELDS) - keys)
            unknown = sorted(keys - set(cls._FIELDS))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError(
                "inference_revision fields are invalid: " + "; ".join(details)
            )
        return cls(
            **{
                field: _revision_token(value[field], f"inference_revision.{field}")
                for field in cls._FIELDS
            }
        )

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in self._FIELDS}

    @property
    def digest(self) -> str:
        return stable_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    """Immutable host snapshot used by every read belonging to one request.

    ``message_upper_bound`` is the largest message row that was eligible when the
    snapshot was captured.  Reads must satisfy both the strict valid-time cutoff
    and this transaction-order bound.  The current request message is therefore
    excluded even when several messages share a one-second platform timestamp.
    """

    snapshot_id: str
    umo: str
    scope_sha256: str
    cutoff_at: int
    message_upper_bound: int
    request_source_key: str
    sender_participant_key: str
    reply_source_key: str
    query_sha256: str
    context_sha256: str
    data_revision: DataRevisionVector
    inference_revision: InferenceRevisionVector
    captured_at: int

    @classmethod
    def from_value(cls, value: object) -> RequestSnapshot:
        if not isinstance(value, Mapping):
            raise ValueError("scope_snapshot must be an object")
        allowed = {
            "snapshot_id",
            "umo",
            "scope_sha256",
            "cutoff_at",
            "message_upper_bound",
            "request_source_key",
            "sender_participant_key",
            "reply_source_key",
            "query_sha256",
            "context_sha256",
            "data_revision",
            "inference_revision",
            "captured_at",
        }
        keys = {str(key) for key in value}
        if keys != allowed:
            missing = sorted(allowed - keys)
            unknown = sorted(keys - allowed)
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError("scope_snapshot fields are invalid: " + "; ".join(details))

        snapshot_id = _bounded_text(
            value.get("snapshot_id"), "scope_snapshot.snapshot_id", limit=160
        )
        if not _SNAPSHOT_ID_RE.fullmatch(snapshot_id):
            raise ValueError("scope_snapshot.snapshot_id is invalid")
        umo = _bounded_text(value.get("umo"), "scope_snapshot.umo", limit=1000)
        scope_sha256 = _sha256(
            value.get("scope_sha256"), "scope_snapshot.scope_sha256"
        )
        expected_scope = hashlib.sha256(umo.encode("utf-8")).hexdigest()
        if scope_sha256 != expected_scope:
            raise ValueError("scope_snapshot.scope_sha256 does not match umo")
        cutoff_at = int(value.get("cutoff_at") or 0)
        captured_at = int(value.get("captured_at") or 0)
        message_upper_bound = int(value.get("message_upper_bound") or 0)
        if cutoff_at <= 0 or captured_at <= 0:
            raise ValueError("scope_snapshot times must be positive")
        if message_upper_bound < 0:
            raise ValueError("scope_snapshot.message_upper_bound must be non-negative")
        return cls(
            snapshot_id=snapshot_id,
            umo=umo,
            scope_sha256=scope_sha256,
            cutoff_at=cutoff_at,
            message_upper_bound=message_upper_bound,
            request_source_key=_bounded_text(
                value.get("request_source_key"),
                "scope_snapshot.request_source_key",
                limit=1000,
                required=False,
            ),
            sender_participant_key=_bounded_text(
                value.get("sender_participant_key"),
                "scope_snapshot.sender_participant_key",
                limit=256,
                required=False,
            ),
            reply_source_key=_bounded_text(
                value.get("reply_source_key"),
                "scope_snapshot.reply_source_key",
                limit=1000,
                required=False,
            ),
            query_sha256=_sha256(
                value.get("query_sha256"), "scope_snapshot.query_sha256"
            ),
            context_sha256=_sha256(
                value.get("context_sha256"), "scope_snapshot.context_sha256"
            ),
            data_revision=DataRevisionVector.from_value(value.get("data_revision")),
            inference_revision=InferenceRevisionVector.from_value(
                value.get("inference_revision")
            ),
            captured_at=captured_at,
        )

    @classmethod
    def create(
        cls,
        *,
        snapshot_id: str,
        umo: str,
        cutoff_at: int,
        message_upper_bound: int,
        request_source_key: str,
        sender_participant_key: str,
        reply_source_key: str,
        query: str,
        context: object,
        data_revision: DataRevisionVector,
        inference_revision: InferenceRevisionVector,
        captured_at: int,
    ) -> RequestSnapshot:
        normalized_query = " ".join(str(query).casefold().split())
        value = {
            "snapshot_id": snapshot_id,
            "umo": umo,
            "scope_sha256": hashlib.sha256(str(umo).encode("utf-8")).hexdigest(),
            "cutoff_at": int(cutoff_at),
            "message_upper_bound": int(message_upper_bound),
            "request_source_key": request_source_key,
            "sender_participant_key": sender_participant_key,
            "reply_source_key": reply_source_key,
            "query_sha256": hashlib.sha256(
                normalized_query.encode("utf-8")
            ).hexdigest(),
            "context_sha256": stable_sha256(context),
            "data_revision": data_revision.as_dict(),
            "inference_revision": inference_revision.as_dict(),
            "captured_at": int(captured_at),
        }
        return cls.from_value(value)

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "umo": self.umo,
            "scope_sha256": self.scope_sha256,
            "cutoff_at": self.cutoff_at,
            "message_upper_bound": self.message_upper_bound,
            "request_source_key": self.request_source_key,
            "sender_participant_key": self.sender_participant_key,
            "reply_source_key": self.reply_source_key,
            "query_sha256": self.query_sha256,
            "context_sha256": self.context_sha256,
            "data_revision": self.data_revision.as_dict(),
            "inference_revision": self.inference_revision.as_dict(),
            "captured_at": self.captured_at,
        }

    @property
    def digest(self) -> str:
        return stable_sha256(self.as_dict())

    def allows_evidence(
        self,
        *,
        umo: str,
        sent_at: int,
        message_row_id: int,
        source_key: str = "",
    ) -> bool:
        """Return the host visibility decision for one immutable source row."""

        if str(umo) != self.umo:
            return False
        if int(sent_at) >= self.cutoff_at:
            return False
        if int(message_row_id) <= 0 or int(message_row_id) > self.message_upper_bound:
            return False
        if source_key and str(source_key) == self.request_source_key:
            return False
        return True

    def same_data_view(self, other: RequestSnapshot) -> bool:
        return (
            self.umo == other.umo
            and self.cutoff_at == other.cutoff_at
            and self.message_upper_bound == other.message_upper_bound
            and self.data_revision == other.data_revision
        )
