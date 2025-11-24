# llm_client.py
"""
Lightweight LLM summarizer helper (OpenAI new syntax, 2024+).

Public API:
    summarize_samples(samples, max_words=5, model="gpt-4o-mini", temperature=0.0, call_llm_fn=None)
      -> (label_str, raw_response_dict_or_error)

Notes:
- samples: list[str] (prefer masked/normalized_text upstream)
- call_llm_fn: optional callable(prompt: str, temperature: float) -> dict-like or str
- This file will try to auto-load a .env in the current working directory using python-dotenv.
"""

from typing import List, Tuple, Optional, Any
import os
import re
import pathlib


# ✅ PUT THIS BLOCK RIGHT HERE (after imports, before OpenAI import)
# ----------------------------------------------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv

    # always load .env FROM THE SAME FOLDER as this script (Windows safe)
    script_dir = pathlib.Path(__file__).resolve().parent
    dotenv_path = script_dir / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f".env loaded from: {dotenv_path}")   # can comment out later
    else:
        print(".env NOT FOUND next to llm_client.py")

except Exception as e:
    print("dotenv load error:", e)
# ----------------------------------------------------------------
# -------------------------
# Prompt + postprocessing
# -------------------------
_PROMPT_TEMPLATE = """You are given N SMS examples from the same SMS campaign, presented highest-similarity first.

Examples:
{examples}

Task: produce a single concise label for this campaign using at most {max_words} words.
- Keep it short, descriptive, and generic (no personal data, no phone numbers, no OTPs).
- Use Title Case or lower-case (consistent across campaigns).
- Do NOT include punctuation at the end.
- Reply with the label only (no explanation).
- Don't use the word "campaign" in the label.
"""

def _build_prompt(samples: List[str], max_words: int) -> str:
    examples_text = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(samples))
    return _PROMPT_TEMPLATE.format(examples=examples_text, max_words=max_words)

_RE_NON_WORD = re.compile(r"[^\w\s\-]", flags=re.UNICODE)

def _postprocess_label(text: str, max_words: int) -> str:
    if not text:
        return ""
    txt = str(text).strip()
    txt = _RE_NON_WORD.sub("", txt)
    tokens = [t for t in txt.split() if t.strip()]
    tokens = tokens[:max_words]  # enforce max length
    return " ".join(tokens).strip()

# -------------------------
# Lazy OpenAI loader
# -------------------------
def _get_openai_client() -> Optional[Any]:
    """Lazy import and instantiate OpenAI. Returns client or None."""
    try:
        from openai import OpenAI  # NEW syntax
        return OpenAI()            # reads OPENAI_API_KEY from env
    except Exception:
        return None

# -------------------------
# Public function
# -------------------------
def summarize_samples(
    samples: List[str],
    max_words: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    call_llm_fn=None,
) -> Tuple[str, Optional[dict]]:
    """Summarize the list of sample messages into a concise campaign label."""
    samples = [str(s).strip() for s in samples if str(s).strip()]
    if not samples:
        return "", None

    prompt = _build_prompt(samples, max_words=max_words)

    # Optional override (Anthropic/local/etc.)
    if call_llm_fn:
        try:
            raw = call_llm_fn(prompt=prompt, temperature=temperature)
            txt = raw.get("text") if isinstance(raw, dict) else str(raw)
            return _postprocess_label(txt, max_words=max_words), raw # type: ignore
        except Exception as e:
            return "", {"error": "call_llm_fn_failed", "exc": str(e)}

    # Try OpenAI (only if API key is set)
    
    if os.getenv("OPENAI_API_KEY"):
        client = _get_openai_client()
        if client:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a concise summarizer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(temperature),
                    max_tokens=32,
                )
                txt = resp.choices[0].message.content
                return _postprocess_label(txt, max_words=max_words), resp
            except Exception as e:
                return "", {"error": "openai_call_failed", "exc": str(e)}
        else:
            return "", {"error": "openai_client_import_failed"}

    return "", {"error": "no_llm_available"}


# ----------------------------------------------------------
# CLI TEST RUNNER  (runs sample messages)
# ----------------------------------------------------------
if __name__ == "__main__":
    import json, textwrap

    print("\n[ llm_client.py self-test (verbose) ]\n")

    example_samples = [
        "use code {OTP} to complete your login at {url:example.com} thanks.",
        "one-time password {OTP} for your account. never share this code.",
        "{name}, use code {OTP} to complete your login at {url:example.com} thanks.",
        "{name}, security code {OTP} to confirm your transaction at {url:example.com}",
        "your otp is {OTP}. do not share. visit {url:example.com}",
        "{name}, your otp is {OTP}. do not share. visit {url:example.com} thanks.",
        "otp {OTP} valid for {NUM} minutes. use it to sign in."
    ]

    label, raw = summarize_samples(example_samples, max_words=5, model="gpt-4o-mini")

    print("Result label:", repr(label))
    print("\nRaw response (repr):")
    print(repr(raw))
    print("\nRaw response (pretty JSON if possible):")
    try:
        print(json.dumps(raw, indent=2, default=str))
    except Exception:
        print(textwrap.indent(str(raw), "  "))

    if not label:
        print("\nHINT: label is empty. Common causes:")
        print(" - OPENAI_API_KEY not set or .env not loaded.")
        print(" - openai package not installed in this Python interpreter.")
        print(" - client import failed or model call failed (see raw response above).")
    print("\nDone.\n")
