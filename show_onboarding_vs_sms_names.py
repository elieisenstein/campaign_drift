#!/usr/bin/env python
import sys
import threading
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import scrolledtext
import tkinter.font as tkfont

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
BASE_DIR = Path(".")

CAMPAIGNS_PATH = BASE_DIR / "data" / "data_raw_input" / "2025-11-13" / "CAMPAIGNS.csv"
STAGE1_OUTPUT_DIR = BASE_DIR / "data" / "stage1" / "output"
ONBOARDING_PATH = BASE_DIR / "data" / "output_onboarding" / "onboarding_names.csv"

OUTPUT_COMPARISON_DIR = BASE_DIR / "data" / "output_comparison"
LLM_RESULTS_CSV = OUTPUT_COMPARISON_DIR / "llm_drift_results.csv"

# Column names
BUSINESS_NUMBER_COL = "business_number"
CAMPAIGN_NAME_COL = "campaign_name"
TEXT_SAMPLE_COL = "text_sample"

# Onboarding columns
ONBOARDING_NAME_COL = "onboarding_name"   # 1a
ONBOARDING_CAMPAIGN_COL = "campaign_name" # 1b

# LLM config (adapt to your environment)
LLM_PROVIDER = "openai"          # or "azure", etc.
LLM_MODEL = "gpt-4o-mini"        # adapt as needed
LLM_TEMPERATURE = 0.0

# -------------------------------------------------------------------
# THEME (toggle here)
# -------------------------------------------------------------------
DARK_MODE = True

DARK_BG = "#121212"        # blackish
DARK_PANEL = "#1B1B1B"     # panel bg
DARK_BORDER = "#2A2A2A"    # subtle border
DARK_TEXT =   "#A8BBE0"            #B7C7D9"      # gray-blue text
DARK_HEADER = "#8FB3D9"    # header text
DARK_SELECT_BG = "#2D4A66" # selection blue-gray
DARK_SELECT_FG = "#FFFFFF"

LIGHT_HEADER = "#0A1A66"   # your existing navy header

# Global cache for LLM results: originator -> result_text
LLM_CACHE: Dict[str, str] = {}

# Adjust the module name/path if different in your project
from compare_onboarding_clusters_llm import compare_onboarding_clusters_llm


# -------------------------------------------------------------------
# DATA LOADING HELPERS
# -------------------------------------------------------------------
def load_campaigns_df() -> pd.DataFrame:
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
    """Originators list (strings) filtered to >4 digits."""
    originators = [str(o) for o in df_campaigns[BUSINESS_NUMBER_COL].dropna().unique().tolist()]
    originators = [o for o in originators if len(o) > 4]
    return sorted(originators)


def load_onboarding_names_for_originator(df_onboarding: pd.DataFrame, originator: str) -> List[str]:
    """
    1a: preserve CSV row order (synced with 1b).
    """
    df_sub = df_onboarding[df_onboarding[BUSINESS_NUMBER_COL].astype(str) == str(originator)]
    if df_sub.empty:
        return [f"[No onboarding rows found for originator {originator}]"]
    return df_sub[ONBOARDING_NAME_COL].dropna().astype(str).tolist()


def load_onboarding_campaign_names_for_originator(df_onboarding: pd.DataFrame, originator: str) -> List[str]:
    """
    1b: preserve CSV row order (synced with 1a).
    """
    df_sub = df_onboarding[df_onboarding[BUSINESS_NUMBER_COL].astype(str) == str(originator)]
    if df_sub.empty:
        return [f"[No onboarding rows found for originator {originator}]"]
    return df_sub[ONBOARDING_CAMPAIGN_COL].dropna().astype(str).tolist()


def load_sms_names(originator: str) -> List[str]:
    """
    2a from ./data/stage1/output/<originator>_campaigns.csv
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
    2b from ./data/stage1/output/<originator>_campaign_examples.csv
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
        return [f"[No examples found for campaign '{campaign_name}']"]

    return df[TEXT_SAMPLE_COL].dropna().astype(str).tolist()


# -------------------------------------------------------------------
# TKINTER APP
# -------------------------------------------------------------------
class CampaignDriftApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Campaign Drift Viewer")
        self.root.geometry("1200x800")

        # Fonts
        ui_font = tkfont.nametofont("TkDefaultFont")
        ui_font.configure(size=14)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=14)

        self.content_font = tkfont.Font(family="Arial", size=14)
        self.header_font = tkfont.Font(family="Arial", size=16, weight="bold")

        # Apply theme early (before widgets)
        self._apply_theme()

        # Data
        self.df_campaigns = load_campaigns_df()
        self.df_onboarding = load_onboarding_df()
        self.originators = load_originators(self.df_campaigns)
        if not self.originators:
            messagebox.showerror("Error", "No originators found in CAMPAIGNS.csv")
            root.destroy()
            return

        self.originator_var = tk.StringVar(value=self.originators[0])

        # LLM cache flags
        self.llm_cache_loaded_from_file: bool = False
        self.prepare_thread: Optional[threading.Thread] = None
        self.prepare_cancelled: bool = False
        self.prepare_running: bool = False

        self._load_llm_cache_from_csv()

        # Build UI
        self._build_widgets()

        # Select first originator
        self.listbox_originators.selection_set(0)
        self.listbox_originators.activate(0)

        first = self.originators[0]
        self.refresh_for_originator(first)
        self._update_llm_panel_for_originator(first)

    # -----------------------------
    # THEME
    # -----------------------------
    def _apply_theme(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        if not DARK_MODE:
            # Default light mode; keep standard ttk look
            return

        # Root window background
        self.root.configure(bg=DARK_BG)

        # ttk base
        style.configure(".", background=DARK_BG, foreground=DARK_TEXT)
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT)

        # LabelFrame
        style.configure("TLabelframe", background=DARK_PANEL, bordercolor=DARK_BORDER)
        style.configure("TLabelframe.Label", background=DARK_PANEL, foreground=DARK_HEADER)

        # Buttons
        style.configure("TButton", background=DARK_PANEL, foreground=DARK_TEXT, bordercolor=DARK_BORDER)
        style.map(
            "TButton",
            background=[("active", DARK_BORDER), ("disabled", DARK_PANEL)],
            foreground=[("active", DARK_TEXT), ("disabled", "#6F7B86")],
        )

        # Scrollbars
        style.configure(
            "TScrollbar",
            background=DARK_PANEL,
            troughcolor=DARK_BG,
            bordercolor=DARK_BORDER,
            arrowcolor=DARK_TEXT,
        )

    def _style_listbox(self, lb: tk.Listbox) -> None:
        if not DARK_MODE:
            return
        lb.configure(
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            selectbackground=DARK_SELECT_BG,
            selectforeground=DARK_SELECT_FG,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_BORDER,
        )

    def _style_text(self, txt: scrolledtext.ScrolledText) -> None:
        if not DARK_MODE:
            return
        txt.configure(
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            insertbackground=DARK_TEXT,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_BORDER,
        )

    # -----------------------------
    # LLM CACHE LOAD/SAVE
    # -----------------------------
    def _load_llm_cache_from_csv(self) -> None:
        global LLM_CACHE
        LLM_CACHE.clear()

        if not LLM_RESULTS_CSV.exists():
            self.llm_cache_loaded_from_file = False
            return

        try:
            df = pd.read_csv(LLM_RESULTS_CSV)
            if BUSINESS_NUMBER_COL in df.columns and "llm_result" in df.columns:
                for _, row in df.iterrows():
                    originator = str(row[BUSINESS_NUMBER_COL])
                    result = str(row["llm_result"])
                    LLM_CACHE[originator] = result
                self.llm_cache_loaded_from_file = True
            else:
                self.llm_cache_loaded_from_file = False
        except Exception:
            self.llm_cache_loaded_from_file = False

    def _save_llm_cache_to_csv(self) -> None:
        global LLM_CACHE
        if not LLM_CACHE:
            return
        OUTPUT_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        rows = [{BUSINESS_NUMBER_COL: o, "llm_result": r} for o, r in LLM_CACHE.items()]
        pd.DataFrame(rows).to_csv(LLM_RESULTS_CSV, index=False)

    # -----------------------------
    # UI BUILD
    # -----------------------------
    def _build_widgets(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # 50/50 columns
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=0)  # LLM band
        main_frame.rowconfigure(1, weight=1)  # columns

        # -------------------------
        # Top: LLM anomalies panel
        # -------------------------
        top_llm_frame = ttk.Frame(main_frame)
        top_llm_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        top_llm_frame.columnconfigure(0, weight=0)  # button area
        top_llm_frame.columnconfigure(1, weight=1)  # anomalies text

        buttons_frame = ttk.Frame(top_llm_frame)
        buttons_frame.grid(row=0, column=0, sticky="nw", padx=(0, 10))

        # Only show "Prepare all" if results file does not exist / wasn't loaded
        self.btn_llm_prepare_all = None
        if not self.llm_cache_loaded_from_file:
            self.btn_llm_prepare_all = ttk.Button(
                buttons_frame,
                text="Prepare all LLM anomalies",
                command=self.on_llm_prepare_all_clicked,
            )
            self.btn_llm_prepare_all.grid(row=0, column=0, sticky="w")

        frame_llm = ttk.LabelFrame(top_llm_frame)
        header_fg = DARK_HEADER if DARK_MODE else LIGHT_HEADER
        frame_llm.configure(labelwidget=tk.Label(
            frame_llm,
            text="LLM Anomalies",
            font=self.header_font,
            fg=header_fg,
            bg=(DARK_PANEL if DARK_MODE else None),
        ))
        frame_llm.grid(row=0, column=1, sticky="nsew")
        frame_llm.rowconfigure(0, weight=1)
        frame_llm.columnconfigure(0, weight=1)

        self.text_llm = scrolledtext.ScrolledText(
            frame_llm,
            wrap="word",
            state="disabled",
            font=self.content_font,
            height=6,
        )
        self.text_llm.grid(row=0, column=0, sticky="nsew")
        self._style_text(self.text_llm)

        # Tags for output
        self.text_llm.config(state="normal")
        if DARK_MODE:
            self.text_llm.tag_configure("ok", foreground="#7EE787")       # soft green
            self.text_llm.tag_configure("anomaly", foreground="#FF7B72")  # soft red
            self.text_llm.tag_configure("info", foreground=DARK_TEXT)
            self.text_llm.tag_configure("progress", foreground=DARK_TEXT)
        else:
            self.text_llm.tag_configure("ok", foreground="#008000")
            self.text_llm.tag_configure("anomaly", foreground="#CC0000")
            self.text_llm.tag_configure("info", foreground="#000000")
            self.text_llm.tag_configure("progress", foreground="#000000")
        self.text_llm.delete("1.0", tk.END)
        self.text_llm.config(state="disabled")

        # -------------------------
        # Bottom: left/right columns (balanced)
        # -------------------------
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        left_frame.rowconfigure(0, weight=1)   # Originators
        left_frame.rowconfigure(1, weight=1)   # 1a
        left_frame.rowconfigure(2, weight=1)   # 1b
        left_frame.columnconfigure(0, weight=1)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        right_frame.rowconfigure(0, weight=1)  # 2a
        right_frame.rowconfigure(1, weight=1)  # 2b
        right_frame.columnconfigure(0, weight=1)

        # Originators list
        frame_originators = ttk.LabelFrame(left_frame)
        frame_originators.configure(labelwidget=tk.Label(
            frame_originators,
            text="Originators",
            font=self.header_font,
            fg=header_fg,
            bg=(DARK_PANEL if DARK_MODE else None),
        ))
        frame_originators.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        frame_originators.rowconfigure(0, weight=1)
        frame_originators.columnconfigure(0, weight=1)

        self.listbox_originators = tk.Listbox(
            frame_originators,
            exportselection=False,
            font=self.content_font,
            width=18,
        )
        self._style_listbox(self.listbox_originators)
        scrollbar_orig = ttk.Scrollbar(frame_originators, orient="vertical", command=self.listbox_originators.yview)
        self.listbox_originators.config(yscrollcommand=scrollbar_orig.set)
        self.listbox_originators.grid(row=0, column=0, sticky="nsew")
        scrollbar_orig.grid(row=0, column=1, sticky="ns")

        for o in self.originators:
            self.listbox_originators.insert(tk.END, o)

        self.listbox_originators.bind("<<ListboxSelect>>", self.on_originator_selected)

        # 1a
        frame_1a = ttk.LabelFrame(left_frame)
        frame_1a.configure(labelwidget=tk.Label(
            frame_1a, text="Onboarding Names", font=self.header_font, fg=header_fg, bg=(DARK_PANEL if DARK_MODE else None)
        ))
        frame_1a.grid(row=1, column=0, sticky="nsew", pady=(0, 5))
        frame_1a.rowconfigure(0, weight=1)
        frame_1a.columnconfigure(0, weight=1)

        self.listbox_1a = tk.Listbox(
            frame_1a,
            exportselection=False,
            font=self.content_font,
            width=42,
        )
        self._style_listbox(self.listbox_1a)
        scrollbar_1a = ttk.Scrollbar(frame_1a, orient="vertical", command=self.listbox_1a.yview)
        self.listbox_1a.config(yscrollcommand=scrollbar_1a.set)
        self.listbox_1a.grid(row=0, column=0, sticky="nsew")
        scrollbar_1a.grid(row=0, column=1, sticky="ns")

        # 1b
        frame_1b = ttk.LabelFrame(left_frame)
        frame_1b.configure(labelwidget=tk.Label(
            frame_1b, text="Onboarding Campaign Names", font=self.header_font, fg=header_fg, bg=(DARK_PANEL if DARK_MODE else None)
        ))
        frame_1b.grid(row=2, column=0, sticky="nsew", pady=(5, 0))
        frame_1b.rowconfigure(0, weight=1)
        frame_1b.columnconfigure(0, weight=1)

        self.listbox_1b = tk.Listbox(
            frame_1b,
            exportselection=False,
            font=self.content_font,
            width=42,
        )
        self._style_listbox(self.listbox_1b)
        scrollbar_1b = ttk.Scrollbar(frame_1b, orient="vertical", command=self.listbox_1b.yview)
        self.listbox_1b.config(yscrollcommand=scrollbar_1b.set)
        self.listbox_1b.grid(row=0, column=0, sticky="nsew")
        scrollbar_1b.grid(row=0, column=1, sticky="ns")

        # 2a
        frame_2a = ttk.LabelFrame(right_frame)
        frame_2a.configure(labelwidget=tk.Label(
            frame_2a, text="SMS Names", font=self.header_font, fg=header_fg, bg=(DARK_PANEL if DARK_MODE else None)
        ))
        frame_2a.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        frame_2a.rowconfigure(0, weight=1)
        frame_2a.columnconfigure(0, weight=1)

        self.listbox_2a = tk.Listbox(
            frame_2a,
            exportselection=False,
            font=self.content_font,
            width=42,
        )
        self._style_listbox(self.listbox_2a)
        scrollbar_2a = ttk.Scrollbar(frame_2a, orient="vertical", command=self.listbox_2a.yview)
        self.listbox_2a.config(yscrollcommand=scrollbar_2a.set)
        self.listbox_2a.grid(row=0, column=0, sticky="nsew")
        scrollbar_2a.grid(row=0, column=1, sticky="ns")
        self.listbox_2a.bind("<<ListboxSelect>>", self.on_sms_name_selected)

        # 2b
        frame_2b = ttk.LabelFrame(right_frame)
        frame_2b.configure(labelwidget=tk.Label(
            frame_2b, text="SMS Examples", font=self.header_font, fg=header_fg, bg=(DARK_PANEL if DARK_MODE else None)
        ))
        frame_2b.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        frame_2b.rowconfigure(0, weight=1)
        frame_2b.columnconfigure(0, weight=1)

        self.text_2b = scrolledtext.ScrolledText(
            frame_2b,
            wrap="word",
            state="disabled",
            font=self.content_font,
            height=10,
        )
        self.text_2b.grid(row=0, column=0, sticky="nsew")
        self._style_text(self.text_2b)

    # -----------------------------
    # UI helpers
    # -----------------------------
    def _update_listbox(self, listbox: tk.Listbox, items: List[str]) -> None:
        listbox.delete(0, tk.END)
        for item in items:
            listbox.insert(tk.END, item)

    def _set_text_2b(self, text: str) -> None:
        self.text_2b.config(state="normal")
        self.text_2b.delete("1.0", tk.END)
        if text:
            self.text_2b.insert(tk.END, text)
        self.text_2b.config(state="disabled")

    def _set_text_llm(self, text: str, tag: str = "info") -> None:
        self.text_llm.config(state="normal")
        self.text_llm.delete("1.0", tk.END)
        if text:
            self.text_llm.insert(tk.END, text, tag)
        self.text_llm.config(state="disabled")

    def _append_text_llm(self, text: str, tag: str = "progress") -> None:
        self.text_llm.config(state="normal")
        self.text_llm.insert(tk.END, text, tag)
        self.text_llm.see(tk.END)
        self.text_llm.config(state="disabled")

    @staticmethod
    def _infer_status_from_result(result_text: str) -> str:
        lower = result_text.lower()
        if "anomaly detected" in lower or "anomal" in lower:
            return "anomaly"
        return "ok"

    def refresh_for_originator(self, originator: str) -> None:
        onboarding_names = load_onboarding_names_for_originator(self.df_onboarding, originator)
        onboarding_campaigns = load_onboarding_campaign_names_for_originator(self.df_onboarding, originator)
        sms_names = load_sms_names(originator)

        self._update_listbox(self.listbox_1a, onboarding_names)
        self._update_listbox(self.listbox_1b, onboarding_campaigns)
        self._update_listbox(self.listbox_2a, sms_names)
        self._set_text_2b("")

    def _update_llm_panel_for_originator(self, originator: str) -> None:
        if originator in LLM_CACHE:
            txt = LLM_CACHE[originator]
            self._set_text_llm(txt, self._infer_status_from_result(txt))
            return

        if self.llm_cache_loaded_from_file:
            self._set_text_llm(
                f"No precomputed LLM result found for originator {originator} in:\n{LLM_RESULTS_CSV}",
                "info",
            )
            return

        self._set_text_llm(
            "LLM anomalies are not prepared yet.\n\n"
            "Click 'Prepare all LLM anomalies' to compute and save them.\n"
            f"Output file will be:\n{LLM_RESULTS_CSV}",
            "info",
        )

    # -----------------------------
    # Events
    # -----------------------------
    def on_originator_selected(self, event: tk.Event) -> None:
        selection = self.listbox_originators.curselection()
        if not selection:
            return
        idx = selection[0]
        originator = self.listbox_originators.get(idx)

        self.originator_var.set(originator)
        self.refresh_for_originator(originator)
        self._update_llm_panel_for_originator(originator)

    def on_sms_name_selected(self, event: tk.Event) -> None:
        selection = self.listbox_2a.curselection()
        if not selection:
            return
        idx = selection[0]
        campaign_name = self.listbox_2a.get(idx)
        originator = self.originator_var.get()

        examples = load_sms_examples(originator, campaign_name)
        self._set_text_2b("\n\n-----\n\n".join(examples))

    # -----------------------------
    # Prepare all (threaded)
    # -----------------------------
    def on_llm_prepare_all_clicked(self) -> None:
        if self.prepare_running:
            return

        self.prepare_running = True
        self.prepare_cancelled = False

        # Disable button immediately
        if self.btn_llm_prepare_all is not None:
            self.btn_llm_prepare_all.config(state="disabled")

        self._set_text_llm("Preparing LLM drift for all originators...\n", "info")

        self.prepare_thread = threading.Thread(target=self._prepare_all_worker, daemon=True)
        self.prepare_thread.start()

    def _prepare_all_worker(self) -> None:
        """
        Runs in background thread.
        Uses root.after(...) to safely update UI.
        """
        global LLM_CACHE

        total = len(self.originators)
        prepared = 0
        skipped_missing = 0
        skipped_empty = 0
        errors = 0

        def ui_append(msg: str) -> None:
            self.root.after(0, lambda: self._append_text_llm(msg, "progress"))

        def ui_set_done(msg: str) -> None:
            self.root.after(0, lambda: self._set_text_llm(msg, "info"))

        ui_append(f"Total originators: {total}\n\n")

        for i, originator in enumerate(self.originators, start=1):
            if self.prepare_cancelled:
                ui_append("\nCancelled.\n")
                break

            # Already cached?
            if originator in LLM_CACHE:
                prepared += 1
                if i % 10 == 0:
                    ui_append(f"Progress: {i}/{total} (cached)\n")
                continue

            # Load inputs
            onboarding_names = load_onboarding_names_for_originator(self.df_onboarding, originator)
            sms_names = load_sms_names(originator)

            # Missing SMS campaigns file
            if sms_names and sms_names[0].startswith("[Missing file:"):
                skipped_missing += 1
                if i % 10 == 0:
                    ui_append(f"Progress: {i}/{total} (missing SMS file)\n")
                continue

            onboarding_names = [x for x in onboarding_names if not x.startswith("[")]
            sms_names = [x for x in sms_names if not x.startswith("[")]

            if not onboarding_names or not sms_names:
                skipped_empty += 1
                if i % 10 == 0:
                    ui_append(f"Progress: {i}/{total} (empty)\n")
                continue

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
                LLM_CACHE[originator] = result_text
                prepared += 1
            except Exception:
                errors += 1

            # UI progress update every 5 originators
            if i % 5 == 0:
                ui_append(
                    f"Progress: {i}/{total} | prepared={prepared} missing={skipped_missing} empty={skipped_empty} errors={errors}\n"
                )

        # Save results
        try:
            self._save_llm_cache_to_csv()
            self.llm_cache_loaded_from_file = True
        except Exception as e:
            ui_append(f"\nFailed saving CSV: {e}\n")

        # Hide button after successful preparation
        def finalize_ui():
            if self.btn_llm_prepare_all is not None:
                self.btn_llm_prepare_all.grid_remove()
                self.btn_llm_prepare_all = None

            # Show current originator result now that cache exists
            current = self.originator_var.get()
            if current:
                self._update_llm_panel_for_originator(current)

        self.root.after(0, finalize_ui)

        done_msg = (
            "\nDone.\n"
            f"Prepared: {prepared}\n"
            f"Skipped (missing SMS file): {skipped_missing}\n"
            f"Skipped (empty inputs): {skipped_empty}\n"
            f"Errors: {errors}\n"
            f"Saved to:\n{LLM_RESULTS_CSV}\n"
        )
        ui_set_done(done_msg)

        # Mark no longer running
        self.prepare_running = False


def main() -> None:
    root = tk.Tk()
    root.state("zoomed")  # Windows: maximized
    CampaignDriftApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
