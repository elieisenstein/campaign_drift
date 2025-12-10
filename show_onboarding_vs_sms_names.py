#!/usr/bin/env python
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext
import tkinter.font as tkfont
import os

# LLM compare function
from comapare_onboarding_clusters_llm import compare_onboarding_clusters_llm

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
BASE_DIR = Path(".")

CAMPAIGNS_PATH = BASE_DIR / "data" / "data_raw_input" / "2025-11-13" / "CAMPAIGNS.csv"
STAGE1_OUTPUT_DIR = BASE_DIR / "data" / "stage1" / "output"
ONBOARDING_PATH = BASE_DIR / "data" / "output_onboarding" / "onboarding_names.csv"

# Column names
BUSINESS_NUMBER_COL = "business_number"
CAMPAIGN_NAME_COL = "campaign_name"
TEXT_SAMPLE_COL = "text_sample"

# Onboarding columns
ONBOARDING_NAME_COL = "onboarding_name"   # 1a
ONBOARDING_CAMPAIGN_COL = "campaign_name" # 1b

# LLM settings (adjust as needed)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")
LLM_MODEL = "gpt-5.1"          # e.g. "gpt-5.1", "gpt-5.1-mini", "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# Runtime cache (singleton) – per originator
LLM_CACHE: Dict[str, str] = {}


# -------------------------------------------------------------------
# DATA LOADING HELPERS
# -------------------------------------------------------------------
def load_campaigns_df() -> pd.DataFrame:
    """Load the master CAMPAIGNS.csv file."""
    if not CAMPAIGNS_PATH.exists():
        messagebox.showerror("Error", f"CAMPAIGNS file not found:\n{CAMPAIGNS_PATH.resolve()}")
        sys.exit(1)

    df = pd.read_csv(CAMPAIGNS_PATH)
    if BUSINESS_NUMBER_COL not in df.columns or CAMPAIGN_NAME_COL not in df.columns:
        messagebox.showerror(
            "Error",
            f"Expected columns '{BUSINESS_NUMBER_COL}' and '{CAMPAIGN_NAME_COL}' "
            f"were not found in {CAMPAIGNS_PATH}",
        )
        sys.exit(1)
    return df


def load_onboarding_df() -> pd.DataFrame:
    """Load onboarding_names.csv for 1a and 1b."""
    if not ONBOARDING_PATH.exists():
        messagebox.showerror("Error", f"Onboarding file not found:\n{ONBOARDING_PATH.resolve()}")
        sys.exit(1)

    df = pd.read_csv(ONBOARDING_PATH)
    missing = [
        col
        for col in (BUSINESS_NUMBER_COL, ONBOARDING_NAME_COL, ONBOARDING_CAMPAIGN_COL)
        if col not in df.columns
    ]
    if missing:
        messagebox.showerror(
            "Error",
            "Onboarding file missing expected columns:\n"
            + ", ".join(missing)
            + f"\nFile: {ONBOARDING_PATH}",
        )
        sys.exit(1)

    return df


def load_originators(df_campaigns: pd.DataFrame) -> List[str]:
    """Return sorted list of originators from CAMPAIGNS."""
    originators = [str(o) for o in df_campaigns[BUSINESS_NUMBER_COL].dropna().unique().tolist()]
    # filter out short codes
    return sorted([o for o in originators if len(o) > 4])


# ---------- 1a & 1b from onboarding_names.csv ----------
def load_onboarding_names_for_originator(df_onboarding: pd.DataFrame, originator: str) -> List[str]:
    """1a: onboarding_name values for the chosen originator."""
    df_sub = df_onboarding[df_onboarding[BUSINESS_NUMBER_COL].astype(str) == str(originator)]
    if df_sub.empty:
        return [f"[No onboarding rows found for originator {originator}]"]

    vals = df_sub[ONBOARDING_NAME_COL].dropna().astype(str).unique().tolist()
    return vals


def load_onboarding_campaign_names_for_originator(df_onboarding: pd.DataFrame, originator: str) -> List[str]:
    """1b: campaign_name values for the chosen originator."""
    df_sub = df_onboarding[df_onboarding[BUSINESS_NUMBER_COL].astype(str) == str(originator)]
    if df_sub.empty:
        return [f"[No onboarding rows found for originator {originator}]"]

    vals = df_sub[ONBOARDING_CAMPAIGN_COL].dropna().astype(str).unique().tolist()
    return vals


# ---------- 2a & 2b from stage1/output ----------
def load_sms_names(originator: str) -> List[str]:
    """
    2a: Load SMS campaign names from:
      ./data/stage1/output/<originator>_campaigns.csv
    """
    path = STAGE1_OUTPUT_DIR / f"{originator}_campaigns.csv"
    if not path.exists():
        return [f"[Missing file: {path.name}]"]

    df = pd.read_csv(path)
    if CAMPAIGN_NAME_COL not in df.columns:
        return [f"[Missing column '{CAMPAIGN_NAME_COL}' in {path.name}]"]

    vals = df[CAMPAIGN_NAME_COL].dropna().astype(str).unique().tolist()
    return sorted(vals)


def load_sms_examples(originator: str, campaign_name: Optional[str]) -> List[str]:
    """
    2b: Load SMS examples from:
      ./data/stage1/output/<originator>_campaign_examples.csv

    Filter rows where campaign_name == selected campaign (from 2a).
    """
    path = STAGE1_OUTPUT_DIR / f"{originator}_campaign_examples.csv"
    if not path.exists():
        return [f"[Missing file: {path.name}]"]

    df = pd.read_csv(path)
    for col in (CAMPAIGN_NAME_COL, TEXT_SAMPLE_COL):
        if col not in df.columns:
            return [f"[Missing column '{col}' in {path.name}]"]

    if campaign_name is not None:
        df = df[df[CAMPAIGN_NAME_COL].astype(str) == str(campaign_name)]

    if df.empty:
        if campaign_name is None:
            return ["[No examples found]"]
        else:
            return [f"[No examples found for campaign '{campaign_name}']"]

    examples = df[TEXT_SAMPLE_COL].dropna().astype(str).tolist()
    return examples


# -------------------------------------------------------------------
# TKINTER APP
# -------------------------------------------------------------------
class CampaignDriftApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Campaign Drift Viewer")
        self.root.geometry("1200x800")

        # -----------------------------
        # Fonts: headers 16, content 14
        # -----------------------------
        ui_font = tkfont.nametofont("TkDefaultFont")
        ui_font.configure(size=14)

        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=14)

        self.content_font = tkfont.Font(family="Arial", size=14)
        self.header_font = tkfont.Font(family="Arial", size=16, weight="bold")

        # Load data
        self.df_campaigns = load_campaigns_df()
        self.df_onboarding = load_onboarding_df()
        self.originators = load_originators(self.df_campaigns)

        if not self.originators:
            messagebox.showerror("Error", "No originators found in CAMPAIGNS.csv")
            root.destroy()
            return

        self.originator_var = tk.StringVar(value=self.originators[0])

        # LLM anomalies panel member
        self.text_llm = None  # type: ignore

        self._build_widgets()
        self.refresh_for_originator(self.originators[0])

    # -------------------------------------------------------------------
    # UI BUILD
    # -------------------------------------------------------------------
    def _build_widgets(self) -> None:
        # Top: Originator selector
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Label(top_frame, text="Originator:", font=self.content_font).grid(row=0, column=0, sticky="w")
        self.originator_combo = ttk.Combobox(
            top_frame,
            textvariable=self.originator_var,
            values=self.originators,
            state="readonly",
            width=20,
            font=self.content_font,
        )
        self.originator_combo.grid(row=0, column=1, sticky="w", padx=(5, 20))
        self.originator_combo.bind("<<ComboboxSelected>>", self.on_originator_changed)

        # Main area: 2 columns
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Layout: row 0 -> columns; row 1 -> LLM panel + button
        main_frame.columnconfigure(0, weight=1)   # left
        main_frame.columnconfigure(1, weight=1)   # right
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)

        # Left column (Onboarding)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)

        # Right column (SMS)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # 1a - Onboarding Names
        frame_1a = ttk.LabelFrame(left_frame)
        frame_1a.configure(labelwidget=tk.Label(
            frame_1a,
            text="Onboarding Names",
            font=self.header_font,
            fg="#0A1A66"
        ))
        frame_1a.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        frame_1a.rowconfigure(0, weight=1)
        frame_1a.columnconfigure(0, weight=1)

        self.listbox_1a = tk.Listbox(
            frame_1a,
            exportselection=False,
            font=self.content_font,
            width=45
        )
        scrollbar_1a = ttk.Scrollbar(frame_1a, orient="vertical", command=self.listbox_1a.yview)
        self.listbox_1a.config(yscrollcommand=scrollbar_1a.set)
        self.listbox_1a.grid(row=0, column=0, sticky="nsew")
        scrollbar_1a.grid(row=0, column=1, sticky="ns")

        # 1b - Onboarding Campaign Names
        frame_1b = ttk.LabelFrame(left_frame)
        frame_1b.configure(labelwidget=tk.Label(
            frame_1b,
            text="Onboarding Campaign Names",
            font=self.header_font,
            fg="#0A1A66"
        ))
        frame_1b.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        frame_1b.rowconfigure(0, weight=1)
        frame_1b.columnconfigure(0, weight=1)

        self.listbox_1b = tk.Listbox(
            frame_1b,
            exportselection=False,
            font=self.content_font,
            width=45
        )
        scrollbar_1b = ttk.Scrollbar(frame_1b, orient="vertical", command=self.listbox_1b.yview)
        self.listbox_1b.config(yscrollcommand=scrollbar_1b.set)
        self.listbox_1b.grid(row=0, column=0, sticky="nsew")
        scrollbar_1b.grid(row=0, column=1, sticky="ns")

        # 2a - SMS Names
        frame_2a = ttk.LabelFrame(right_frame)
        frame_2a.configure(labelwidget=tk.Label(
            frame_2a,
            text="Messages' Names",
            font=self.header_font,
            fg="#0A1A66"
        ))
        frame_2a.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        frame_2a.rowconfigure(0, weight=1)
        frame_2a.columnconfigure(0, weight=1)

        self.listbox_2a = tk.Listbox(
            frame_2a,
            exportselection=False,
            font=self.content_font,
            width=45
        )
        scrollbar_2a = ttk.Scrollbar(frame_2a, orient="vertical", command=self.listbox_2a.yview)
        self.listbox_2a.config(yscrollcommand=scrollbar_2a.set)
        self.listbox_2a.grid(row=0, column=0, sticky="nsew")
        scrollbar_2a.grid(row=0, column=1, sticky="ns")
        self.listbox_2a.bind("<<ListboxSelect>>", self.on_sms_name_selected)

        # 2b - SMS Examples
        frame_2b = ttk.LabelFrame(right_frame)
        frame_2b.configure(labelwidget=tk.Label(
            frame_2b,
            text="Messages' Examples",
            font=self.header_font,
            fg="#0A1A66"
        ))
        frame_2b.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        frame_2b.rowconfigure(0, weight=1)
        frame_2b.columnconfigure(0, weight=1)

        self.text_2b = scrolledtext.ScrolledText(
            frame_2b,
            wrap="word",
            state="disabled",
            font=self.content_font,
            width=45
        )
        self.text_2b.grid(row=0, column=0, sticky="nsew")

        # ---------------------------------------------------------------
        # Row 1 of main_frame: Button + LLM Anomalies panel
        # ---------------------------------------------------------------
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=0)
        bottom_frame.columnconfigure(1, weight=1)

        # Button to trigger LLM check
        self.btn_llm_check = ttk.Button(
            bottom_frame,
            text="Check Consistency with LLM",
            command=self.on_llm_check_clicked
        )
        self.btn_llm_check.grid(row=0, column=0, sticky="w", padx=(0, 10))

        # LLM Anomalies panel
        frame_llm = ttk.LabelFrame(bottom_frame)
        frame_llm.configure(labelwidget=tk.Label(
            frame_llm,
            text="LLM Anomalies",
            font=self.header_font,
            fg="#0A1A66"
        ))
        frame_llm.grid(row=0, column=1, sticky="nsew")
        frame_llm.rowconfigure(0, weight=1)
        frame_llm.columnconfigure(0, weight=1)

        self.text_llm = scrolledtext.ScrolledText(
            frame_llm,
            wrap="word",
            state="disabled",
            font=self.content_font,
            height=4
        )
        self.text_llm.grid(row=0, column=0, sticky="nsew")

        # Define tags for colored output
        self.text_llm.config(state="normal")
        self.text_llm.tag_configure("ok", foreground="#008000")       # green
        self.text_llm.tag_configure("anomaly", foreground="#CC0000")  # red
        self.text_llm.tag_configure("info", foreground="#000000")     # black
        self.text_llm.delete("1.0", tk.END)
        self.text_llm.config(state="disabled")

    # -------------------------------------------------------------------
    # REFRESH / HELPERS
    # -------------------------------------------------------------------
    def refresh_for_originator(self, originator: str) -> None:
        """Update all 4 panels when originator changes."""
        onboarding_names = load_onboarding_names_for_originator(self.df_onboarding, originator)
        onboarding_campaigns = load_onboarding_campaign_names_for_originator(
            self.df_onboarding, originator
        )
        sms_names = load_sms_names(originator)

        self._update_listbox(self.listbox_1a, onboarding_names)
        self._update_listbox(self.listbox_1b, onboarding_campaigns)
        self._update_listbox(self.listbox_2a, sms_names)
        self._set_text_2b("")
        self._set_text_llm("", "info")  # clear LLM result panel

    def _update_listbox(self, listbox: tk.Listbox, items: List[str]) -> None:
        listbox.delete(0, tk.END)
        for item in items:
            listbox.insert(tk.END, item)

    def _set_text_2b(self, text: str) -> None:
        self.text_2b.config(state="normal")
        self.text_2b.delete("1.0", tk.END)
        self.text_2b.insert(tk.END, text)
        self.text_2b.config(state="disabled")

    def _set_text_llm(self, text: str, status: str = "info") -> None:
        """Update the LLM anomalies panel with colored text."""
        if self.text_llm is None:
            return
        self.text_llm.config(state="normal")
        self.text_llm.delete("1.0", tk.END)

        tag = "info"
        if status == "ok":
            tag = "ok"
        elif status == "anomaly":
            tag = "anomaly"

        self.text_llm.insert(tk.END, text, (tag,))
        self.text_llm.config(state="disabled")

    def _get_listbox_items(self, listbox: tk.Listbox) -> List[str]:
        return listbox.get(0, tk.END)

    # -------------------------------------------------------------------
    # EVENT HANDLERS
    # -------------------------------------------------------------------
    def on_originator_changed(self, event: tk.Event) -> None:
        originator = self.originator_var.get()
        if originator:
            self.refresh_for_originator(originator)

    def on_sms_name_selected(self, event: tk.Event) -> None:
        """When clicking a campaign name in 2a, show its examples in 2b."""
        selection = self.listbox_2a.curselection()
        if not selection:
            return

        index = selection[0]
        campaign_name = self.listbox_2a.get(index)
        originator = self.originator_var.get()

        examples = load_sms_examples(originator, campaign_name)
        text = "\n\n-----\n\n".join(examples)
        self._set_text_2b(text)

    def on_llm_check_clicked(self) -> None:
        """Run LLM consistency check for current originator (on demand)."""
        originator = self.originator_var.get()
        if not originator:
            messagebox.showwarning("Warning", "No originator selected.")
            return

        # Get 1a and 2a values
        onboarding_names_raw = list(self._get_listbox_items(self.listbox_1a))
        sms_names_raw = list(self._get_listbox_items(self.listbox_2a))

        # Filter out placeholder lines that start with '[' (like "[Missing file...]")
        onboarding_names = [x for x in onboarding_names_raw if not x.startswith("[")]
        sms_names = [x for x in sms_names_raw if not x.startswith("[")]

        if not onboarding_names:
            self._set_text_llm(
                "Cannot run LLM check: no valid onboarding names (1a) for this originator.",
                "info",
            )
            return

        if not sms_names:
            self._set_text_llm(
                "Cannot run LLM check: no valid SMS names (2a) for this originator.",
                "info",
            )
            return

        # Use cache if available
        global LLM_CACHE
        if originator in LLM_CACHE:
            cached_text = LLM_CACHE[originator]
            status = self._infer_status_from_result(cached_text)
            self._set_text_llm(cached_text, status)
            return

        # Build dataframes for the LLM compare function
        df_onboarding = pd.DataFrame({"name": onboarding_names})
        df_clusters = pd.DataFrame({"name": sms_names})

        try:
            result_text, _raw = compare_onboarding_clusters_llm(
                df_onboarding=df_onboarding,
                df_clusters=df_clusters,
                provider=LLM_PROVIDER,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                onboarding_col="name",
                cluster_col="name",
            )
        except Exception as e:
            err_msg = f"Error calling LLM: {e}"
            self._set_text_llm(err_msg, "info")
            return

        # Cache the textual result for this originator
        LLM_CACHE[originator] = result_text

        status = self._infer_status_from_result(result_text)
        self._set_text_llm(result_text, status)

    def _infer_status_from_result(self, text_result: str) -> str:
        """
        Infer status ('ok' / 'anomaly' / 'info') from the LLM text.
        This is heuristic, based on your prompt contract.
        """
        normalized = (text_result or "").lower()
        if "all clustered names fit the onboarding name" in normalized:
            return "ok"
        if "anomaly detected" in normalized:
            return "anomaly"
        return "info"


def main() -> None:
    root = tk.Tk()
    CampaignDriftApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
