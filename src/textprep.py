"""Priprema teksta: normalizacija naslova oglasa."""
import re
import unicodedata

_CYR = dict(zip(
    "абвгдђежзијклљмнњопрстћуфхцчџш",
    ["a", "b", "v", "g", "d", "dj", "e", "z", "z", "i", "j", "k", "l", "lj", "m",
     "n", "nj", "o", "p", "r", "s", "t", "c", "u", "f", "h", "c", "c", "dz", "s"],
))
_SPECIAL = {"đ": "dj", "š": "s", "č": "c", "ć": "c", "ž": "z"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Mala slova, ćirilica -> latinica, bez dijakritika, samo slova i cifre."""
    text = str(text).lower()
    out = []
    for ch in text:
        if ch in _CYR:
            out.append(_CYR[ch])
        elif ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        else:
            out.append(ch)
    text = "".join(out)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _NON_ALNUM.sub(" ", text)
    return text.strip()


# Varijante pripreme teksta koje se porede u svesci 02 (ulaz je uvek originalni naslov iz kolone `name`)
_NON_WORD = re.compile(r"[\W_]+")   # sve što nije slovo ili cifra (slova sa kvačicama ostaju)

VARIANTS = {
    "bez normalizacije": lambda s: str(s),
    "samo mala slova": lambda s: str(s).lower(),
    "mala slova + bez interpunkcije": lambda s: _NON_WORD.sub(" ", str(s).lower()).strip(),
    "puna normalizacija": normalize,
}
