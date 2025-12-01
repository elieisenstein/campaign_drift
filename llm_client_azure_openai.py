# llm_client.py
"""
Lightweight LLM summarizer helper (uses AzureChatClient API from your working example).

Public API:
    summarize_samples(samples, max_words=5, model="gpt-4o-mini", temperature=0.0, call_llm_fn=None)
      -> (label_str, raw_response_dict_or_error)

Notes:
- samples: list[str] (prefer masked/normalized_text upstream)
- call_llm_fn: optional callable(prompt: str, temperature: float) -> dict-like or str
- This file will try to auto-load a .env in the same folder as this script using python-dotenv.
"""

from typing import List, Tuple, Optional, Any
import os
import re
import pathlib
import sys

# ✅ PUT THIS BLOCK RIGHT HERE (after imports, before Azure import)
# ----------------------------------------------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv

    script_dir = pathlib.Path(__file__).resolve().parent
    dotenv_path = script_dir / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f".env loaded from: {dotenv_path}")
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
- If language is not English, give the label in English and add at the end of the name the language in parentheses, e.g. (Spanish)!!
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
# Lazy Azure client loader (uses your working example's API)
# -------------------------
def _get_azure_chat_client() -> Optional[Any]:
    """Lazy import and instantiate AzureChatClient via ServiceConfig like your working example."""
    try:
        # these imports should match the module that provides ServiceConfig and AzureChatClient
        script_dir = pathlib.Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
            
        from client import ServiceConfig, AzureChatClient
        cfg = ServiceConfig.load()
        return AzureChatClient(cfg)
    except Exception as e:
        print(f"AzureChatClient import/instantiation error: {e}")
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
            return _postprocess_label(txt, max_words=max_words), raw  # type: ignore
        except Exception as e:
            return "", {"error": "call_llm_fn_failed", "exc": str(e)}

    # Try AzureChatClient (from your working example)
    client = _get_azure_chat_client()
    if client:
        try:
            # call the same chat(...) API as in your working example
            results, meta = client.chat(prompt)
            # results might be a string or dict; try to extract text
            if isinstance(results, dict):
                # common keys: "text", "content", etc.
                txt = results.get("text") or results.get("content") or str(results)
            else:
                txt = str(results)

            return _postprocess_label(txt, max_words=max_words), {"azure_raw": meta, "azure_result": results}
        except Exception as e:
            return "", {"error": "azure_chat_call_failed", "exc": str(e)}
    else:
        return "", {"error": "azure_client_import_failed"}

# ----------------------------------------------------------
# CLI TEST RUNNER  (runs sample messages)
# ----------------------------------------------------------
if __name__ == "__main__":
    import json, textwrap

    print("\n[ llm_client.py self-test (verbose) ]\n")

    example_samples = [
        #"use code {OTP} to complete your login at {url:example.com} thanks.",
        #"one-time password {OTP} for your account. never share this code.",
        #"{name}, use code {OTP} to complete your login at {url:example.com} thanks.",
        #"{name}, security code {OTP} to confirm your transaction at {url:example.com}",
        #"your otp is {OTP}. do not share. visit {url:example.com}",
        #"{name}, your otp is {OTP}. do not share. visit {url:example.com} thanks.",
        #"otp {OTP} valid for {NUM} minutes. use it to sign in."
        "cuenta chase {OTP}: transacciÃ³n de tarjeta de dÃ©bito de ${NUM} a costco whse #{OTP} el nov {NUM}, {OTP} a las {TIME} hora del este, excede ${NUM}.",
        "cuenta chase {OTP}: transacciÃ³n de tarjeta de dÃ©bito de ${NUM} a klarna*{url:temu.com} el nov {NUM}, {OTP} a las {TIME} hora del este, excede ${NUM}.",
        "cuenta chase {OTP}: transacciÃ³n de tarjeta de dÃ©bito de ${NUM} a sunpass el nov {NUM}, {OTP} a las {TIME} hora del este, excede ${NUM}.",
        "cuenta chase {OTP}: transacciÃ³n de tarjeta de dÃ©bito de ${NUM} a roadrunner express el nov {NUM}, {OTP} a las {TIME} hora del este, excede ${NUM}.",
        "cuenta chase {OTP}: transacciÃ³n de tarjeta de dÃ©bito de ${NUM} a {url:amazon.com} el nov {NUM}, {OTP} a las {TIME} hora del este, excede ${NUM}."
    ]

    #label, raw = summarize_samples(example_samples, max_words=5, model="gpt-4o-mini")
    label, raw = summarize_samples(example_samples)

    print("Result label:", repr(label))
    #print("\nRaw response (repr):")
    """
    print(repr(raw))
    print("\nRaw response (pretty JSON if possible):")
    try:
        print(json.dumps(raw, indent=2, default=str))
    except Exception:
        print(textwrap.indent(str(raw), "  "))

    if not label:
        print("\nHINT: label is empty. Common causes:")
        print(" - ServiceConfig.load() may fail or .env not set for Azure client.")
        print(" - client module not installed or AzureChatClient API changed.")
        """
    print("\nDone.\n")
