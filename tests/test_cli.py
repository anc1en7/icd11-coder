import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from coder.__main__ import fill, finished, read_notes  # noqa: E402
from coder.search import affirmed  # noqa: E402


class Cli(unittest.TestCase):
    def test_affirmed_drops_ruled_out(self):
        kept = affirmed("Dx viral URTI, malaria excluded. No cough. ?typhoid fever")
        self.assertIn("URTI", kept)
        self.assertNotIn("malaria", kept)
        self.assertNotIn("cough", kept)
        self.assertNotIn("typhoid", kept)

    def test_fill_drops_invalid_and_pads(self):
        self.assertEqual(fill(["BAD", "A"], {"A", "B", "C"}, ["B", "C"]), ["A", "B", "C"])

    def test_finished_removes_half_line(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p = os.path.join(d, "o.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"id": "a", "codes": ["X"]}\n{"id": "b", "co')
            self.assertEqual(finished(p), {"a"})
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"id": "a", "codes": ["X"]}\n')

    def test_read_notes_survives_bad_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p = os.path.join(d, "i.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"id": "a", "text": "x"}\nnot json\n\n{"id": "a", "text": "dup"}\n{"id": "b"}\n')
            self.assertEqual(read_notes(p), [("a", "x"), ("b", "")])

    def test_every_id_gets_valid_codes_on_odd_notes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            src, dst = os.path.join(d, "i.jsonl"), os.path.join(d, "o.jsonl")
            notes = [{"id": "empty", "text": ""}, {"id": "none", "text": None}, {"id": "junk", "text": "### 123"},
                     {"id": "long", "text": "fever " * 50000}, {"id": "fr", "text": "Paludisme à Plasmodium falciparum"}]
            with open(src, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(n) + "\n" for n in notes)
            env = dict(os.environ)
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODER_DATA"):
                env.pop(key, None)
            subprocess.run([sys.executable, "-m", "coder", src, dst], cwd=ROOT, env=env,
                           check=True, capture_output=True)
            with gzip.open(os.path.join(ROOT, "data", "who", "codes.jsonl.gz"), "rt", encoding="utf-8") as f:
                valid = {json.loads(line)["code"] for line in f}
            with open(dst, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f]
            self.assertEqual([r["id"] for r in rows], [n["id"] for n in notes])
            for r in rows:
                self.assertEqual(len(r["codes"]), 3)
                self.assertTrue(set(r["codes"]) <= valid, r)


if __name__ == "__main__":
    unittest.main()
