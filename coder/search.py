"""Local code search: BM25 over WHO text, plus exact matches of whole WHO terms."""
import collections
import gzip
import json
import math
import re

from .text import tokens

K1, B = 1.2, 0.3
MAX_PHRASE = 8
PHRASE_WEIGHT = 3.0
ASSESSMENT_WEIGHT = 3
TERM_CAP = 60

ASSESSMENT = re.compile(r"^\s*(a\s*[:/]|a&p|assessment|impression|imp\s*[:.]|dx|diagnos[ie]s|diagnostic|"
                        r"final dx|working dx)", re.I | re.M)
CLAUSE = re.compile(r"[.;\n,]+")
NEGATED_CLAUSE = re.compile(r"\b(excluded|ruled out|r/o|unlikely|neg|negative)\b", re.I)
NEGATOR = re.compile(r"\b(no|not|denies|denied|without|nil)\b.*", re.I)
QUERIED = re.compile(r"\?\s*[^\W\d_]+")
HEAD = re.compile(r"[(;]")
POOL = 300
LIST_SIZE, FAMILY_OF = 30, 10


def affirmed(text):
    """Drop clauses and phrases the note rules out or only queries."""
    kept = []
    for clause in CLAUSE.split(text):
        if NEGATED_CLAUSE.search(clause):
            continue
        kept.append(NEGATOR.sub(" ", QUERIED.sub(" ", clause)))
    return " . ".join(kept)


def assessment_lines(text):
    return " ".join(line for line in text.splitlines() if ASSESSMENT.match(line))


def fuse(weighted_lists, k=20):
    """Reciprocal rank fusion over (weight, codes) pairs."""
    scores = collections.Counter()
    for weight, codes in weighted_lists:
        for i, code in enumerate(codes):
            scores[code] += weight / (k + i + 1)
    return [code for code, _ in scores.most_common()]


def read_lines(read):
    """Split the MAIN, ALT and OTHER lines of a model reading into a dict of lists."""
    fields = collections.defaultdict(list)
    for line in read.splitlines():
        key, sep, value = line.partition(":")
        if sep and value.strip():
            fields[key.strip().upper()].append(value.strip())
    return fields


def short_name(text):
    return HEAD.split(text)[0].strip()


class Index:
    def __init__(self, codes_path, shorthand_path):
        self.records = {}
        with gzip.open(codes_path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.records[r["code"]] = r
        self.valid = set(self.records)
        self.children = collections.defaultdict(list)
        for code, r in self.records.items():
            if r["leaf"] and r.get("parent_code"):
                self.children[r["parent_code"]].append(code)
        with open(shorthand_path, encoding="utf-8") as f:
            terms = json.load(f)["terms"]
        self.shorthand = [(re.compile(rf"(?<![\w]){re.escape(k)}(?![\w])"), v)
                          for k, v in sorted(terms.items(), key=lambda kv: -len(kv[0]))]
        self.postings = collections.defaultdict(list)
        self.phrases = collections.defaultdict(set)
        self.lengths = {}
        for code, r in self.records.items():
            if not r["leaf"]:
                continue
            names = [r["title"], *r.get("terms", [])[:TERM_CAP], *r.get("inclusions", [])]
            translated = list(r.get("i18n", {}).values())
            words = tokens(r["title"]) * 3
            for name in (*names[1:], *translated):
                words += tokens(name)
            for w, tf in collections.Counter(words).items():
                self.postings[w].append((code, tf))
            self.lengths[code] = len(words)
            for name in (*names, *translated):
                key = tuple(tokens(name, keep_stop=True))
                if 0 < len(key) <= MAX_PHRASE:
                    self.phrases[key].add(code)
        self.avg_len = sum(self.lengths.values()) / len(self.lengths)
        n = len(self.lengths)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

    def expand(self, text):
        extra = [v for pattern, v in self.shorthand if pattern.search(text)]
        return text + " . " + " . ".join(extra) if extra else text

    def search(self, text, limit=50, clean=False):
        """clean=True skips negation handling, for model output that is already affirmed."""
        text = text or ""
        check = (lambda t: t) if clean else affirmed
        kept = check(self.expand(text))
        assessment = check(self.expand(assessment_lines(text)))
        query = collections.Counter(tokens(kept))
        for w in tokens(assessment):
            query[w] += ASSESSMENT_WEIGHT
        scores = collections.Counter()
        for w, qtf in query.items():
            idf = self.idf.get(w)
            if idf is None:
                continue
            weight = idf * (1 + math.log(qtf))
            for code, tf in self.postings[w]:
                norm = K1 * (1 - B + B * self.lengths[code] / self.avg_len)
                scores[code] += weight * tf * (K1 + 1) / (tf + norm)
        for source, boost in ((kept, 1.0), (assessment, 2.0)):
            seq = tokens(source, keep_stop=True)
            for i in range(len(seq)):
                for size in range(1, min(MAX_PHRASE, len(seq) - i) + 1):
                    for code in self.phrases.get(tuple(seq[i:i + size]), ()):
                        scores[code] += PHRASE_WEIGHT * boost * size * size
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def codes(self, text, clean=False):
        return [code for code, _ in self.search(text, limit=POOL, clean=clean)]

    def ranked(self, note, read):
        """Merge search on each extracted line with search on the raw note."""
        fields = read_lines(read)
        lists = []
        for key, weight in (("MAIN", 3), ("ALT", 2)):
            for line in fields[key]:
                lists += [(weight, self.codes(line, clean=True)), (weight, self.codes(short_name(line), clean=True))]
        lists += [(1, self.codes(short_name(line), clean=True)) for line in fields["OTHER"]]
        return fuse([(4, fuse(lists)), (1, self.codes(note))])

    def options(self, ranked, size=LIST_SIZE, family_of=FAMILY_OF):
        """The top codes plus the siblings of the best ones, so the model can pick the right level of detail."""
        picked = list(ranked[:size])
        seen = set(picked)
        for code in ranked[:family_of]:
            for sibling in self.children.get(self.records[code].get("parent_code"), ()):
                if sibling not in seen:
                    seen.add(sibling)
                    picked.append(sibling)
        return picked

    def listing(self, codes, detail=False):
        lines = []
        for code in codes:
            r = self.records[code]
            line = f"{code}: {r['title']}"
            if detail:
                also = [*(r.get("inclusions") or []), *(r.get("terms") or [])][:3]
                if also:
                    line += " | also: " + "; ".join(also)
                if r.get("definition"):
                    line += " | " + r["definition"][:140]
            lines.append(line)
        return "\n".join(lines)
