"""Usage: python -m coder <input.jsonl> <output.jsonl>

Writes one line per note as soon as it is ready, so a stopped run keeps its finished notes.
Rerunning with the same output file keeps finished notes and does the rest.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .llm import connect
from .search import Index

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CODER_DATA", os.path.join(HERE, "..", "data"))
# Used only when a note gives search nothing to work with. All are checked against the code list.
DEFAULT_CODES = ("MG26", "CA07.0", "MD90.0")
WORKERS = 8


def read_notes(path):
    notes, seen = [], set()
    with open(path, encoding="utf-8-sig") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                note_id = row["id"]
            except (ValueError, KeyError, TypeError):
                print(f"line {number}: skipped, no readable id", file=sys.stderr)
                continue
            if note_id not in seen:
                seen.add(note_id)
                notes.append((note_id, row.get("text") if isinstance(row.get("text"), str) else ""))
    return notes


def finished(path):
    """Return ids already written, and rewrite the file without any half-written last line."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    good, ids = [], set()
    for line in lines:
        try:
            row = json.loads(line)
            if row["id"] not in ids and row["codes"] and line.endswith("\n"):
                ids.add(row["id"])
                good.append(line)
        except (ValueError, KeyError, TypeError):
            pass
    if len(good) != len(lines):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(good)
    return ids


def fill(codes, valid, defaults):
    out = []
    for code in (*codes, *defaults):
        if code in valid and code not in out:
            out.append(code)
    return out[:3]


def code_note(text, index, llm):
    """Search answers come first, so any model failure only lowers quality for this note."""
    fallback = index.codes(text)
    if not llm or not text.strip():
        return fallback[:3]
    read = llm.read(text)
    if not read:
        return fallback[:3]
    ranked = index.ranked(text, read)
    picks = llm.rank(text, read, index.listing(index.options(ranked)))
    return [*picks, *ranked][:12]


def main(argv):
    if argv[1:] in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    start = time.time()
    index = Index(os.path.join(DATA, "who", "codes.jsonl.gz"), os.path.join(DATA, "shorthand.json"))
    defaults = [c for c in DEFAULT_CODES if c in index.valid]
    defaults += sorted(c for c, r in index.records.items() if r["leaf"])[:3]
    print(f"index ready in {time.time() - start:.1f}s", file=sys.stderr, flush=True)

    notes = read_notes(argv[1])
    out_path = argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = finished(out_path)
    llm = connect()
    print("using " + (llm.describe() if llm else "search only"), file=sys.stderr, flush=True)
    todo = [n for n in notes if n[0] not in done]
    lock = threading.Lock()
    failed = [0]
    with open(out_path, "a", encoding="utf-8") as out:
        def run(note):
            note_id, text = note
            try:
                codes = code_note(text, index, llm)
            except Exception as e:
                with lock:
                    failed[0] += 1
                print(f"{note_id}: failed: {e!r}", file=sys.stderr)
                codes = []
            row = {"id": note_id, "codes": fill(codes, index.valid, defaults)}
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

        # Search alone is fast, and one worker keeps its output in input order.
        with ThreadPoolExecutor(WORKERS if llm else 1) as pool:
            list(pool.map(run, todo))
    print(f"{len(notes)} notes, {len(done)} already done, {failed[0]} failed, {time.time() - start:.1f}s",
          file=sys.stderr)
    if llm:
        print(llm.report(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
