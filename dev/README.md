# Isolated development workflow

The primary development target is the local machine, not the 2C2G production
server.

Planned shadow runtime:

- Python 3.13 managed by `uv`.
- AstrBot pinned to 4.27.1.
- Independent root under repository-local `.dev/mr-shadow`.
- Dashboard bound to `127.0.0.1:6285`.
- WebChat or a fake adapter only; no NapCat/OneBot configuration.
- No production provider credentials.

The current repository deliberately does not install or start that runtime. The
framework-independent core and replay tests are the first compatibility gate.

Fixture JSONL uses `NormalizedMessage` fields. Production history exports must
be one-way, read-only, and pseudonymized before they are added to this folder.
Real exports and generated databases are ignored by Git; only the synthetic
sample fixture is tracked.

The private retrieval benchmark is documented in
`docs/RETRIEVAL_BENCHMARK.md`. Its generated corpus and direct annotations live
under the repository-level `.dev/benchmarks/` directory, never under this
tracked fixture directory.
