from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

import httpx


CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_chunked(
    *,
    url: str,
    output: Path,
    expected_size: int,
    expected_sha256: str,
    proxy: str | None,
    chunk_bytes: int,
    retries: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    chunk_path = output.with_name(output.name + ".chunk")
    if output.exists():
        if output.stat().st_size == expected_size and sha256_file(output) == expected_sha256:
            print(f"already complete: {output}", flush=True)
            return
        raise ValueError(f"existing output is invalid: {output}")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise ValueError(f"partial file is larger than expected: {partial}")

    timeout = httpx.Timeout(connect=30.0, read=45.0, write=45.0, pool=30.0)
    with httpx.Client(
        proxy=proxy,
        follow_redirects=True,
        timeout=timeout,
        headers={"Accept-Encoding": "identity"},
    ) as client:
        while (partial.stat().st_size if partial.exists() else 0) < expected_size:
            start = partial.stat().st_size if partial.exists() else 0
            if start >= expected_size:
                break
            end = min(expected_size - 1, start + chunk_bytes - 1)
            wanted = end - start + 1
            failure: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    received = 0
                    with client.stream(
                        "GET",
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise ValueError(
                                f"server ignored range {start}-{end}: {response.status_code}"
                            )
                        match = CONTENT_RANGE_RE.fullmatch(
                            str(response.headers.get("content-range") or "")
                        )
                        if not match or tuple(map(int, match.groups())) != (
                            start,
                            end,
                            expected_size,
                        ):
                            raise ValueError(
                                f"unexpected content-range: {response.headers.get('content-range')}"
                            )
                        with chunk_path.open("wb") as handle:
                            for block in response.iter_bytes(1024 * 1024):
                                handle.write(block)
                                received += len(block)
                    if received != wanted:
                        raise ValueError(
                            f"short range {start}-{end}: expected={wanted}, got={received}"
                        )
                    with partial.open("ab") as destination, chunk_path.open("rb") as source:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                            destination.write(block)
                    chunk_path.unlink(missing_ok=True)
                    completed = partial.stat().st_size
                    print(
                        f"{completed}/{expected_size} bytes "
                        f"({completed / expected_size:.1%})",
                        flush=True,
                    )
                    failure = None
                    break
                except Exception as exc:  # Network retry boundary.
                    failure = exc
                    chunk_path.unlink(missing_ok=True)
                    if attempt < retries:
                        time.sleep(min(10.0, 0.5 * 2 ** (attempt - 1)))
            if failure is not None:
                raise RuntimeError(f"failed range {start}-{end}") from failure

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"download size mismatch: expected={expected_size}, got={actual_size}")
    actual_sha256 = sha256_file(partial)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"download hash mismatch: expected={expected_sha256}, got={actual_sha256}"
        )
    os.replace(partial, output)
    print(f"verified sha256 and moved to: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one Hugging Face file in retryable byte ranges and verify SHA-256."
    )
    parser.add_argument("repo_id")
    parser.add_argument("filename")
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--proxy")
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    if args.expected_size <= 0 or args.chunk_mib <= 0 or args.retries <= 0:
        raise ValueError("size, chunk size, and retries must be positive")
    expected_sha256 = args.sha256.casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    url = (
        f"https://huggingface.co/{args.repo_id}/resolve/{args.revision}/"
        f"{args.filename}?download=true"
    )
    download_chunked(
        url=url,
        output=args.output.resolve(),
        expected_size=args.expected_size,
        expected_sha256=expected_sha256,
        proxy=args.proxy,
        chunk_bytes=args.chunk_mib * 1024 * 1024,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
