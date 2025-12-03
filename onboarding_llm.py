# onboarding_llm.py

from typing import Any, Tuple
import llm_client_openai as lc_openai
import llm_client_azure_openai as lc_azure


def build_onboarding_prompt(row, max_words: int = 5) -> str:
    """
    Build prompt from onboarding columns B,E,G,K,P,R:
    business_number, description, campaign_name, content_type,
    call_to_action, customer_experience.
    """
    fields = {
        "business_number":      row["business_number"],
        "description":          row["description"],
        "campaign_name":        row["campaign_name"],
        "content_type":         row["content_type"],
        "call_to_action":       row["call_to_action"],
        "customer_experience":  row["customer_experience"],
    }
    examples_text = "\n".join(f"{k}: {v}" for k, v in fields.items())

    prompt = f"""You are given onboarding data for a single SMS/MMS campaign.
The data comes from several columns in the campaign's onboarding form.

Onboarding record:
{examples_text}

Task: produce a single concise onboarding name using at most {max_words} words.
- Keep it short, descriptive, and generic (no personal data, no phone numbers, no OTPs).
- Use Title Case or lower-case (consistent across campaigns).
- Do NOT include punctuation at the end.
- Reply with the name only (no explanation).
- Don't use the word "campaign" or "onboarding" in the name.
"""
    return prompt


def llm_onboarding_label(
    row,
    provider: str = "openai",       # "openai" or "azure"
    max_words: int = 5,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Tuple[str, Any]:
    """
    Use either OpenAI or Azure backend (via the two llm_client_* helpers)
    to generate a short onboarding name for a single row.

    Returns (label, raw_response).
    """
    provider = provider.lower()
    prompt = build_onboarding_prompt(row, max_words=max_words)

    if provider == "openai":
        client = lc_openai._get_openai_client()
        if client is None:
            raise RuntimeError(
                "OpenAI client not available. "
                "Check OPENAI_API_KEY, .env and the openai package."
            )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=32,
        )
        raw_text = resp.choices[0].message.content or ""
        label = lc_openai._postprocess_label(raw_text, max_words=max_words)
        return label, resp

    elif provider == "azure":
        client = lc_azure._get_azure_chat_client()
        if client is None:
            raise RuntimeError(
                "AzureChatClient not available. "
                "Check ServiceConfig.load() and your Azure settings/.env."
            )

        results, meta = client.chat(prompt)
        if isinstance(results, dict):
            raw_text = results.get("text") or results.get("content") or str(results)
        else:
            raw_text = str(results)

        label = lc_azure._postprocess_label(raw_text, max_words=max_words)
        return label, {"azure_raw": meta, "azure_result": results}

    else:
        raise ValueError(f"Unknown provider: {provider!r}")
