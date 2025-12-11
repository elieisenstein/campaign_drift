"""
Compare onboarding campaign names (first group) with clustered names (second group)
using an LLM (OpenAI or Azure).

Supports both GPT-5.x (requires max_completion_tokens) and older models such as
gpt-4o / gpt-4o-mini (which require max_tokens).
"""

from typing import Any, Tuple
import pandas as pd
import os

import llm_client_openai as lc_openai
import llm_client_azure_openai as lc_azure

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")

# ---------------------------------------------------------------------------
# Model Utility
# ---------------------------------------------------------------------------

def _is_gpt_5_model(model_name: str) -> bool:
    """Return True if the model uses the GPT-5 token parameter interface."""
    if not model_name:
        return False
    return model_name.lower().startswith("gpt-5")


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_compare_prompt(
    df_onboarding: pd.DataFrame,
    df_clusters: pd.DataFrame,
    onboarding_col: str = "name",
    cluster_col: str = "name",
) -> str:
    """
    Build an LLM prompt comparing onboarding campaign names (First group)
    and auto-clustered names (Second group). All items are numbered.
    """

    instruction_text = (
        "You are given two groups of names:\n"
        "• First group – the authoritative campaign name(s) provided during onboarding.\n"
        "• Second group – names produced automatically by clustering recent messages.\n\n"
        "Your task: Determine whether every name in the second group can reasonably fit "
        "under the general name(s) of the first group.\n\n"
        "Instructions:\n"
        
        "IMPORTANT – Apply a generous, inclusive interpretation:\n"
        "- Consider broad thematic similarity as sufficient for a match.\n"
        "- Do NOT flag differences in wording, phrasing, specificity, level of detail, or language "
        "  (e.g., Spanish vs. English) as anomalies if the overall subject matter is consistent.\n"
        "- Treat verification codes, reminders, notices, alerts, summaries, digests, or event-related "
        "  messages as belonging to the same general communication category unless they are clearly "
        "  unrelated to the domain of the onboarding name(s).\n"
        "- Only flag an anomaly when a name is clearly outside the scope, meaning it represents a "
        "  genuinely different topic, industry, intent, or communication purpose.\n"
        "- When in doubt, assume the name fits.\n\n"


        
        "- Treat the First group as the reference campaign identity.\n"
        "- For each name in the Second group, assess whether it semantically belongs to, "
        "  or can be considered a subtype of, the First group.\n"
        "- If all second-group names fit under the First group, respond only:\n"
        '  \"All clustered names fit the onboarding name.\".\n'
        "- If any name does not fit, respond:\n"
        '  \"Anomaly detected.\".\n'
        "  Then give a clear, concise explanation for each non-fitting name, describing "
        "why it does not align with the onboarding name(s).\n"
    )

    # Build numbered first group
    first_group_lines = []
    for idx, row in enumerate(df_onboarding.itertuples(index=False), start=1):
        value = getattr(row, onboarding_col)
        first_group_lines.append(f'{idx}. "{value}"')

    # Build numbered second group
    second_group_lines = []
    for idx, row in enumerate(df_clusters.itertuples(index=False), start=1):
        value = getattr(row, cluster_col)
        second_group_lines.append(f"{idx}. {value}")

    first_group_block = "First group:\n" + "\n".join(first_group_lines)
    second_group_block = "Second group:\n" + "\n".join(second_group_lines)

    prompt = f"{instruction_text}\n{first_group_block}\n\n{second_group_block}\n"
    return prompt


# ---------------------------------------------------------------------------
# Core Function
# ---------------------------------------------------------------------------

def compare_onboarding_clusters_llm(
    df_onboarding: pd.DataFrame,
    df_clusters: pd.DataFrame,
    provider: str = "openai",
    model: str = "gpt-5.1",
    temperature: float = 0.0,
    onboarding_col: str = "name",
    cluster_col: str = "name",
) -> Tuple[str, Any]:
    """
    Send a comparison prompt to the selected LLM.

    Returns:
        (text_result, raw_response)
    """

    provider = provider.lower()
    prompt = build_compare_prompt(
        df_onboarding=df_onboarding,
        df_clusters=df_clusters,
        onboarding_col=onboarding_col,
        cluster_col=cluster_col,
    )

    # -------------------------------------------------------------------
    # OpenAI Provider
    # -------------------------------------------------------------------
    if provider == "openai":
        client = lc_openai._get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client not available. Check your OPENAI_API_KEY.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert analyst of SMS/MMS campaign naming consistency. "
                    "Follow the user's instructions exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Decide token parameter based on model family
        if _is_gpt_5_model(model):
            # GPT-5.x models use max_completion_tokens
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_completion_tokens=256,
            )
        else:
            # Older models use max_tokens
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=256,
            )

        text_result = resp.choices[0].message.content or ""
        return text_result, resp

    # -------------------------------------------------------------------
    # Azure Provider
    # -------------------------------------------------------------------
    elif provider == "azure":
        client = lc_azure._get_azure_chat_client()
        if client is None:
            raise RuntimeError(
                "AzureChatClient not available. Check your Azure settings."
            )

        results, meta = client.chat(prompt)
        if isinstance(results, dict):
            text_result = results.get("text") or results.get("content") or str(results)
        else:
            text_result = str(results)

        return text_result, {"azure_raw": meta, "azure_result": results}

    else:
        raise ValueError(f"Unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Main Example (Your Provided Example)
# ---------------------------------------------------------------------------

def main():
    """
    Example usage using your ParentSquare sample.
    """

    df_onboarding = pd.DataFrame(
        {"name": ["ParentSquare School Notifications"]}
    )

    df_clusters = pd.DataFrame(
        {
            "name": [
                "ParentSquare Phone Verification Code",
                "School Bus Delay Notification",
                "Early Dismissal and Schedule Change",
                "School Message Summary Spanish",
                "School Events Digest",
                "Appointment Reminders Spanish",
                "Let's play football this weekend!",
            ]
        }
    )

    # You may change the model to "gpt-5.1-mini" or "gpt-4o-mini"
    provider = LLM_PROVIDER
    model = "gpt-5.1"
    temperature = 0.0

    result_text, _raw = compare_onboarding_clusters_llm(
        df_onboarding=df_onboarding,
        df_clusters=df_clusters,
        provider=provider,
        model=model,
        temperature=temperature,
        onboarding_col="name",
        cluster_col="name",
    )

    print("LLM result:")
    print(result_text)


if __name__ == "__main__":
    main()
