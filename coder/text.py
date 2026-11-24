"""Text normalisation shared by the index and the notes."""
import re
import unicodedata

SPELLING = (("haem", "hem"), ("anaem", "anem"), ("oedema", "edema"), ("oesoph", "esoph"),
            ("diarrhoea", "diarrhea"), ("tumour", "tumor"), ("paed", "ped"), ("foet", "fet"),
            ("ischaem", "ischem"), ("leukaem", "leukem"), ("gynaec", "gynec"))
WORD = re.compile(r"[^\W_]+", re.UNICODE)
CJK = re.compile(r"[㐀-鿿]")
STOP = frozenset("""a an and are as at be by for from has have he her his in is it its of on or she that the
this to was were with due other unspecified specified nos patient""".split())


def stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def normalise(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    for old, new in SPELLING:
        text = text.replace(old, new)
    return text


def tokens(text, keep_stop=False):
    out = []
    for w in WORD.findall(normalise(text)):
        if CJK.search(w):
            out.extend(w[i:i + 2] for i in range(max(len(w) - 1, 1)))
        elif keep_stop or w not in STOP:
            out.append(stem(w))
    return out
