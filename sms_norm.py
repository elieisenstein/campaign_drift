# sms_norm.py
# Requirements: pandas (for the batch helpers). No external hashing deps.
# If you prefer xxhash64 later, see the commented lines in hash64_hex.

import re, unicodedata, hashlib
from typing import Tuple, Iterable
import pandas as pd

# ---------- Confusables & zero-width cleanup ----------
CONFUSABLES_MAP = {
    # Cyrillic → Latin
    "а":"a","е":"e","о":"o","р":"p","с":"c","у":"y","х":"x","і":"i","ј":"j","һ":"h","к":"k","м":"m","н":"n","т":"t","в":"v",
    "А":"A","В":"B","С":"C","Е":"E","Н":"H","К":"K","М":"M","О":"O","Р":"P","Т":"T","Х":"X","І":"I","Ј":"J",
    # Greek → Latin (subset)
    "α":"a","β":"b","γ":"y","δ":"d","ε":"e","η":"n","ι":"i","κ":"k","ν":"v","ο":"o","ρ":"p","τ":"t","υ":"u","χ":"x",
    "Α":"A","Β":"B","Ε":"E","Ζ":"Z","Η":"H","Ι":"I","Κ":"K","Μ":"M","Ν":"N","Ο":"O","Ρ":"P","Τ":"T","Υ":"Y","Χ":"X",
    # Misc
    "ℓ":"l"
}
ZERO_WIDTH = {
    "\u200B","\u200C","\u200D","\uFEFF","\u2060",
    "\u200E","\u200F",
    "\u202A","\u202B","\u202C","\u202D","\u202E"
}

def fold_confusables(s: str) -> str:
    if not s: return s
    out = []
    for ch in s:
        if ch in ZERO_WIDTH:
            continue
        out.append(CONFUSABLES_MAP.get(ch, ch))
    return "".join(out)

def normalize_unicode_basic(s: str) -> str:
    s = fold_confusables(s)
    s = unicodedata.normalize("NFKC", s)
    # collapse 3+ repeats to 2 (punct/emojis)
    s = re.sub(r'(.)\1{2,}', r'\1\1', s)
    s = re.sub(r'\s+', ' ', s, flags=re.MULTILINE).strip()
    return s

# ---------- Patterns ----------
EMAIL_RE = re.compile(r'(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b')
PHONE_RE = re.compile(r'(?x)\b(?:\+?\d[\d\-\.\s()\u00A0]{6,}\d)\b')  # 7+ digits total
URL_RE   = re.compile(r'(?i)\b((?:https?://|www\.)[^\s<>"]+|(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>"]*)?)')

GENERIC_TERMS = {"dear","customer","client","friend","partner","user","member"}

# Fix 1: don’t treat signoffs as leading names (negative look-ahead)
LEADING_NAME_RE = re.compile(
    r'(?mi)^(?!(?:thanks|thank you|cheers|regards|best)\b)'
    r'(?P<name>[A-Z][A-Za-z\'\-\.]{1,20}(?:\s+[A-Z][A-Za-z\'\-\.]{0,20}){0,2})\s*,\s+'
)
# Fix 2: apply signoff replacement FIRST
SIGNOFF_NAME_RE = re.compile(
    r'(?i)(?P<prefix>\b(?:thanks|thank you|cheers|regards|best)\s*,\s*)'
    r'(?P<name>[A-Z][A-Za-z\'\-\.]{1,20})\b'
)

# OTP & numbers (after phone/URL masking)
OTP_CONTEXT_RE = re.compile(
    r'(?ix)\b(?:otp|code|verification\s*code|security\s*code|one[-\s]?time(?:\s*password)?|pin)\b'
    r'[^0-9]{0,20}\b([0-9]{4,8})\b'
)
OTP_BARE_RE   = re.compile(r'\b([0-9]{4,8})\b')
GENERIC_NUM_RE = re.compile(r'(?<![{A-Za-z])\b\d+(?:[\.,]\d+)?\b(?![}\w])')

# Updated gate regex: match "gate" + optional spaces + letter(s) + digits
GATE_RE = re.compile(r"\bgate\s*[a-zA-Z]\d{1,3}\b", re.IGNORECASE)

# ---------- Masking helpers ----------
def _is_generic(name: str) -> bool:
    return name.strip().lower() in GENERIC_TERMS

def anonymize_names_preserving_greeting(text: str) -> str:
    # 1) signoff: "Thanks, Alice" → "Thanks, {NAME}"
    def signoff_repl(m):
        return f"{m.group('prefix')}{{NAME}}"
    t = SIGNOFF_NAME_RE.sub(signoff_repl, text)
    # 2) greeting: "John, ..." → "{NAME}, ..."
    def leading_repl(m):
        name = m.group("name")
        return "{NAME}, " if not _is_generic(name) else m.group(0)
    t = LEADING_NAME_RE.sub(leading_repl, t)
    return t

def replace_emails(text: str) -> Tuple[str, bool]:
    has = bool(EMAIL_RE.search(text))
    return EMAIL_RE.sub("{EMAIL}", text), has

def replace_phones(text: str) -> Tuple[str, bool]:
    # Ensures +1 (212) 555-0199 and 2125550199 both → {PHONE}
    def _repl(m):
        digits = sum(ch.isdigit() for ch in m.group(0))
        return "{PHONE}" if digits >= 7 else m.group(0)
    new_text = PHONE_RE.sub(_repl, text)
    return new_text, (new_text != text)

def _domain_only(url: str) -> str:
    u = url.strip()
    u = re.sub(r'(?i)^(https?://)', '', u)
    dom = u.split('/', 1)[0].split(':', 1)[0]
    if dom.lower().startswith("www."):
        dom = dom[4:]
    try:
        dom = fold_confusables(dom)
        dom = unicodedata.normalize("NFKD", dom).encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    dom = dom.lower()
    parts = [p for p in dom.split('.') if p]
    if len(parts) >= 2:
        dom = '.'.join(parts[-2:])   # crude eTLD+1 fallback
    return dom

def replace_urls_with_domain(text: str) -> Tuple[str, int]:
    count = 0
    def repl(m):
        nonlocal count
        count += 1
        dom = _domain_only(m.group(0))
        return f"{{URL:{dom}}}" if dom else "{URL}"
    return URL_RE.sub(repl, text), count

def mask_otps_and_numbers(text: str) -> str:

    # 0) Gate identifiers: "gate C18" or standalone "C18" → {GATE}
    t = GATE_RE.sub("{GATE}", text)

    # 1) OTP in context: replace digits only
    def ctx_repl(m):
        full = m.group(0); num = m.group(1)
        return full.replace(num, "{OTP}")
    t = OTP_CONTEXT_RE.sub(ctx_repl, t)
    # 2) Bare 4–8 digit tokens likely OTPs
    t = OTP_BARE_RE.sub("{OTP}", t)
    # 3) Remaining numbers → {NUM}
    t = GENERIC_NUM_RE.sub("{NUM}", t)
    return t

# ---------- Public normalization API ----------
def normalize_text(original: str) -> str:
    """
    - Unicode/whitespace cleanup, confusable folding
    - Mask names (greetings/signoffs), emails, phones
    - URLs → {URL:domain}
    - Mask OTPs ({OTP}) and other numbers ({NUM})
    - Lowercase + spacing
    """
    t = normalize_unicode_basic(original)
    t = anonymize_names_preserving_greeting(t)
    t, _ = replace_emails(t)
    t, _ = replace_phones(t)
    t, _ = replace_urls_with_domain(t)
    t = t.lower()
    t = mask_otps_and_numbers(t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---------- 64-bit stable hash for per-window dedupe ----------
def hash64_hex(s: str, seed: int = 0) -> str:
    # Built-in: 64-bit hex via blake2b digest_size=8 (no extra deps).
    h = hashlib.blake2b(digest_size=8, person=str(seed).encode("utf-8"))
    h.update(s.encode("utf-8", "ignore"))
    return h.hexdigest()
    # Prefer xxhash64?
    # import xxhash
    # return xxhash.xxh64(s.encode("utf-8","ignore"), seed=seed).hexdigest()

# ---------- Batch helpers ----------
def normalize_and_hash_series(texts: Iterable[str], seed: int = 0) -> pd.DataFrame:
    df = pd.DataFrame({"raw_text": list(texts)})
    df["normalized_text"] = df["raw_text"].astype(str).map(normalize_text)
    df["template_hash_xx64"] = df["normalized_text"].map(lambda s: hash64_hex(s, seed=seed))
    return df

def dedupe_by_hash(df: pd.DataFrame, hash_col: str = "template_hash_xx64"):
    first = ~df[hash_col].duplicated(keep="first")
    dedup_df = df.loc[first].copy()
    group_sizes = df.groupby(hash_col, as_index=True).size().rename("count_in_window")
    dedup_df = dedup_df.merge(group_sizes, left_on=hash_col, right_index=True, how="left")
    return dedup_df, group_sizes
