"""Host-verified stored summaries with a source closure independent of raw budget."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .narrative_bindings import project_narrative_record
from .snapshot import RequestSnapshot, stable_sha256


STORED_DERIVATION_SCHEMA = "mr-memory.stored-derivation.v1"
MAX_STORED_DERIVATIONS = 64
_KINDS = {"semantic", "episode", "plastic_edge"}
_ROLES = {"USER", "BOT", "SYSTEM", "UNKNOWN"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def stored_derivation_id(value: Mapping[str, object]) -> str:
    return "stored:" + stable_sha256({k: v for k, v in value.items() if k != "derivation_id"})


def validate_stored_derivation(value: object, *, snapshot: RequestSnapshot | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("stored derivation must be an object")
    descriptor = copy.deepcopy(dict(value))
    required = {"schema_version", "derivation_id", "kind", "owner_id", "text", "content_revision",
                "scope", "dependency_keys", "source_fingerprints", "source_roles"}
    optional = {"narrative_bindings", "head_created_at", "head_updated_at"}
    if not required.issubset(descriptor) or set(descriptor) - required - optional:
        raise ValueError("stored derivation fields are invalid")
    if descriptor["schema_version"] != STORED_DERIVATION_SCHEMA or descriptor["kind"] not in _KINDS:
        raise ValueError("stored derivation schema or kind is invalid")
    if not isinstance(descriptor["owner_id"], str) or not descriptor["owner_id"]:
        raise ValueError("stored derivation owner_id must be a nonempty string")
    texts = descriptor["text"]
    if (not isinstance(texts, dict) or not texts
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in texts.items())
            or not any(texts.values())):
        raise ValueError("stored derivation text must contain stored string fields")
    if descriptor["content_revision"] != stable_sha256(texts):
        raise ValueError("stored derivation content revision differs from its text")
    scope = descriptor["scope"]
    if not isinstance(scope, dict) or set(scope) != {"umo", "before_sent_at", "message_upper_bound"}:
        raise ValueError("stored derivation scope is invalid")
    if (not isinstance(scope["umo"], str) or not scope["umo"]
            or type(scope["before_sent_at"]) is not int or scope["before_sent_at"] <= 0
            or type(scope["message_upper_bound"]) is not int or scope["message_upper_bound"] < 0):
        raise ValueError("stored derivation scope bounds are invalid")
    if snapshot is not None and (scope["umo"] != snapshot.umo
            or scope["before_sent_at"] > snapshot.cutoff_at
            or scope["message_upper_bound"] != snapshot.message_upper_bound):
        raise ValueError("stored derivation scope differs from the request snapshot")
    dependencies = descriptor["dependency_keys"]
    fingerprints = descriptor["source_fingerprints"]
    if (not isinstance(dependencies, list) or not dependencies
            or any(not isinstance(key, str) or not key for key in dependencies)
            or len(set(dependencies)) != len(dependencies)
            or not isinstance(fingerprints, dict) or set(dependencies) != set(fingerprints)):
        raise ValueError("stored derivation requires its complete source fingerprints")
    roles = set()
    for key in dependencies:
        fingerprint = fingerprints[key]
        expected = {"message_id", "sent_at", "revision_no", "content_sha256", "role", "umo", "is_deleted"}
        if (not isinstance(fingerprint, dict) or not expected.issubset(fingerprint)
                or set(fingerprint) - expected - {"evidence_roles"}):
            raise ValueError("stored derivation source fingerprint fields are invalid")
        if fingerprint["umo"] != scope["umo"] or fingerprint["is_deleted"] is not False:
            raise ValueError("stored derivation has a deleted or cross-scope source")
        if (type(fingerprint["message_id"]) is not int or fingerprint["message_id"] <= 0
                or fingerprint["message_id"] > scope["message_upper_bound"]
                or type(fingerprint["sent_at"]) is not int or fingerprint["sent_at"] >= scope["before_sent_at"]
                or type(fingerprint["revision_no"]) is not int or fingerprint["revision_no"] <= 0
                or not isinstance(fingerprint["content_sha256"], str)
                or not _SHA256.fullmatch(fingerprint["content_sha256"])):
            raise ValueError("stored derivation source is outside the snapshot or lacks a revision")
        if snapshot is not None and key == snapshot.request_source_key:
            raise ValueError("stored derivation cannot depend on the current request")
        if fingerprint["role"] not in _ROLES:
            raise ValueError("stored derivation has an unknown source role")
        relations = fingerprint.get("evidence_roles", [])
        if (not isinstance(relations, list)
                or any(not isinstance(item, str) or not item for item in relations)
                or len(relations) != len(set(relations))):
            raise ValueError("stored derivation evidence relations are invalid")
        roles.add(fingerprint["role"])
    if descriptor["source_roles"] != sorted(roles):
        raise ValueError("stored derivation source roles differ from its fingerprints")
    if descriptor["derivation_id"] != stored_derivation_id(descriptor):
        raise ValueError("stored derivation content hash is invalid")
    return descriptor


def build_stored_derivation(*, kind: str, owner_id: str, text: Mapping[str, str],
                           scope: Mapping[str, object], source_fingerprints: Mapping[str, object],
                           narrative_bindings: object = None, head_created_at: object = None,
                           head_updated_at: object = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema_version": STORED_DERIVATION_SCHEMA, "kind": kind, "owner_id": str(owner_id),
        "text": copy.deepcopy(dict(text)), "content_revision": stable_sha256(dict(text)),
        "scope": copy.deepcopy(dict(scope)), "dependency_keys": sorted(source_fingerprints),
        "source_fingerprints": copy.deepcopy(dict(source_fingerprints)),
        "source_roles": sorted({item["role"] for item in source_fingerprints.values()}),
    }
    for name, value in (("narrative_bindings", narrative_bindings), ("head_created_at", head_created_at),
                        ("head_updated_at", head_updated_at)):
        if value is not None:
            descriptor[name] = copy.deepcopy(value)
    descriptor["derivation_id"] = stored_derivation_id(descriptor)
    return validate_stored_derivation(descriptor)


def stored_derivation_allowlist(packet: object, *, snapshot: RequestSnapshot) -> dict[str, dict[str, Any]]:
    records = packet.get("stored_derivations", []) if isinstance(packet, Mapping) else []
    if not isinstance(records, list) or len(records) > MAX_STORED_DERIVATIONS:
        raise ValueError("stored derivations catalog must be a bounded array")
    allowlist = {}
    for record in records:
        descriptor = validate_stored_derivation(record, snapshot=snapshot)
        key = descriptor["derivation_id"]
        if key in allowlist:
            raise ValueError("stored derivations catalog contains duplicate IDs")
        allowlist[key] = descriptor
    return allowlist


def stored_derivation_reader_view(descriptor: Mapping[str, object]) -> dict[str, object]:
    record = project_narrative_record({**descriptor["text"],
                                      "narrative_bindings": descriptor.get("narrative_bindings")})
    return {
        "derivation_id": descriptor["derivation_id"], "kind": descriptor["kind"],
        "text": {key: value for key, value in record.items() if key in descriptor["text"]},
        "source_count": len(descriptor["dependency_keys"]), "source_roles": descriptor["source_roles"],
        "basis": "STORED_DERIVATION", "source_closure": "HOST_VERIFIED_COMPLETE",
        "raw_sources_in_packet": "PARTIAL_OR_NOT_EXPANDED",
        **{key: value for key, value in record.items() if key in {
            "narrative_bindings", "reader_text_identity_scope", "reader_withheld_fields"}},
    }


@dataclass(frozen=True, slots=True)
class StoredDerivationEvidence:
    derivation_id: str
    descriptor_json: str

    @classmethod
    def from_value(cls, value: object, *, snapshot: RequestSnapshot) -> StoredDerivationEvidence:
        import json
        descriptor = validate_stored_derivation(value, snapshot=snapshot)
        return cls(descriptor["derivation_id"], json.dumps(descriptor, ensure_ascii=False, sort_keys=True))

    def as_dict(self) -> dict[str, Any]:
        import json
        return json.loads(self.descriptor_json)

    @property
    def dependency_keys(self) -> tuple[str, ...]:
        return tuple(self.as_dict()["dependency_keys"])

    @property
    def source_roles(self) -> tuple[str, ...]:
        return tuple(self.as_dict()["source_roles"])
