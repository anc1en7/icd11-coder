# ICD-11 Coder

ICD-11 Coder is a batch CLI that proposes up to three ranked ICD-11 MMS codes per clinical note. With Python 3.10 or later, you can run it offline against the bundled 2026-01 index with no third-party Python packages.

## Quick start

```sh
git clone https://github.com/anc1en7/icd11-coder-standalone.git
cd icd11-coder-standalone
python -m coder examples/notes.jsonl out/example.jsonl
```

The example contains two fictional notes. Supply your own UTF-8 JSONL file with a unique string `id` and a `text` field on each line.

| File | Example line |
| --- | --- |
| Input | `{"id":"example-1","text":"Assessment: acute cystitis."}` |
| Output | `{"id":"example-1","codes":["GC00.1","1A70.0Y","GC00.Z"]}` |

The output lists the best candidate first. The CLI writes each result as it finishes and skips IDs already present in the output file on a rerun. Choose a new output path when you change notes or settings and want to recompute predictions.

## Docker Compose

```sh
docker compose run --build --rm coder /app/examples/notes.jsonl /out/example.jsonl
```

Put your own file at `in/notes.jsonl` and pass `/in/notes.jsonl` as the first argument. Compose mounts `in/` read-only and writes results to `out/` on the host.

## Optional model-assisted ranking

For a native Python run, install the SDKs with `python -m pip install -r requirements-models.txt`. The Docker image includes them. Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or both in your shell. Docker Compose also reads these variables from a local `.env` file.

Local search matches the note against WHO titles, index terms and inclusions. With a configured provider, the CLI sends note text to the provider to extract diagnoses and rerank the retrieved codes. It falls back to local search when model calls are unavailable or the run reaches its model budget. Use search-only mode when note text needs to stay on your machine. Have a qualified reviewer check suggested codes before using them in clinical or billing workflows.

| Environment variable | Purpose |
| --- | --- |
| `CODER_READ_MODEL`, `CODER_RANK_MODEL` | Try these models before the built-in preferences. |
| `CODER_RANK_EFFORT` | Set OpenAI ranking effort. Defaults to `low`. |
| `CODER_MAX_USD` | Estimated model spend threshold. Defaults to US$4.50. |
| `CODER_MAX_MINUTES` | Time threshold for model calls. Defaults to 50 minutes. |
| `CODER_CALL_TIMEOUT` | SDK request timeout in seconds. Defaults to 60. |

## Data and tests

The bundled index covers 18,375 codes from ICD-11 MMS 2026-01. It includes English index terms and official titles in seven additional languages. See the [WHO data notice](data/who/NOTICE.md) for attribution and licensing.

```sh
python -m unittest discover -s tests -v
```

The tests exercise the offline CLI and use fake clients for model calls.
