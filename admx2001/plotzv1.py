#!/usr/bin/env python3
"""
Impedance Plotter
=================
A small GUI for viewing impedance-analyser sweep data (e.g. EVAL-ADMX2001 CSV
exports).  Load one or more CSV files, pick the runs you want, and switch
between these views:

    * Impedance            R = Re(Z) and X = Im(Z) vs frequency
    * Admittance           G = Re(Y) and B = Im(Y) vs frequency
    * Bode - Impedance     |Z| and phase θ vs frequency
    * Bode - Admittance    |Y| and phase φ vs frequency
    * Cs - Rs              series equivalent circuit   (Z = Rs + 1/(jωCs))
    * Cp - Rp              parallel equivalent circuit (Y = 1/Rp + jωCp)
    * Nyquist              -Im(Z) vs Re(Z)

Expected CSV layout (column names are matched case-insensitively and a few
common aliases are accepted):

    Run_name, Run_id, Requested_Frequency_Hz, R_ohm, X_ohm, Z_mag_ohm, ...

Lines starting with '#' are treated as comments. If R/X columns are missing,
|Z| and θ (degrees) are used instead. If a Run_id / Run_name column exists,
each run becomes its own selectable trace.

Requirements:  pip install numpy pandas matplotlib
Usage:         python impedance_plotter.py [file1.csv file2.csv ...]
"""

import os
import sys

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.ticker import EngFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# --------------------------------------------------------------------------- #
#  Column detection
# --------------------------------------------------------------------------- #
FREQ_NAMES = ["measured_frequency_hz", "frequency_hz", "requested_frequency_hz",
              "freq_hz", "frequency", "freq", "f_hz", "f"]
R_NAMES = ["r_ohm", "r", "real", "re_z", "z_real", "zreal", "z_re"]
X_NAMES = ["x_ohm", "x", "imag", "im_z", "z_imag", "zimag", "z_im"]
ZMAG_NAMES = ["z_mag_ohm", "z_mag", "zmag", "|z|", "z_abs", "impedance"]
THETA_NAMES = ["theta_deg", "theta", "phase_deg", "phase", "z_phase", "theta_rad"]
RUN_ID_NAMES = ["run_id", "runid", "sweep_id"]
RUN_NAME_NAMES = ["run_name", "runname", "run", "sweep", "name"]


def find_column(columns, candidates):
    """Return the first column whose normalised name matches a candidate."""
    lookup = {c.strip().lower(): c for c in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None


def read_csv_flexible(path):
    """Read a CSV that may start with '#' comment lines."""
    df = pd.read_csv(path, comment="#", skipinitialspace=True)
    if df.shape[1] == 1:  # probably not comma separated -> let pandas sniff
        df = pd.read_csv(path, comment="#", sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
#  Data model
# --------------------------------------------------------------------------- #
def derive_quantities(f, Z):
    """Compute every plotted quantity from frequency (Hz) and complex Z (Ω)."""
    w = 2 * np.pi * f
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = 1.0 / Z
        q = {
            "R": Z.real,
            "X": Z.imag,
            "Zmag": np.abs(Z),
            "theta": np.degrees(np.angle(Z)),
            "G": Y.real,
            "B": Y.imag,
            "Ymag": np.abs(Y),
            "phi": np.degrees(np.angle(Y)),
            # Series model:   Z = Rs + 1/(jωCs)   ->  Cs = -1/(ωX)
            "Rs": Z.real,
            "Cs": -1.0 / (w * Z.imag),
            # Parallel model: Y = 1/Rp + jωCp     ->  Rp = 1/G, Cp = B/ω
            "Rp": 1.0 / Y.real,
            "Cp": Y.imag / w,
        }
    return q


class Dataset:
    def __init__(self, name, source, f, Z):
        order = np.argsort(f)
        self.name = name
        self.source = source
        self.f = f[order]
        self.Z = Z[order]
        self.q = derive_quantities(self.f, self.Z)

    def to_frame(self):
        q = self.q
        return pd.DataFrame({
            "Dataset": self.name,
            "Frequency_Hz": self.f,
            "R_ohm": q["R"], "X_ohm": q["X"],
            "Z_mag_ohm": q["Zmag"], "Theta_deg": q["theta"],
            "G_S": q["G"], "B_S": q["B"],
            "Y_mag_S": q["Ymag"], "Phi_deg": q["phi"],
            "Cs_F": q["Cs"], "Rs_ohm": q["Rs"],
            "Cp_F": q["Cp"], "Rp_ohm": q["Rp"],
        })


def load_datasets(path):
    """Parse a CSV file into one Dataset per run."""
    df = read_csv_flexible(path)
    cols = list(df.columns)

    fcol = find_column(cols, FREQ_NAMES)
    rcol = find_column(cols, R_NAMES)
    xcol = find_column(cols, X_NAMES)
    zcol = find_column(cols, ZMAG_NAMES)
    tcol = find_column(cols, THETA_NAMES)

    if fcol is None:
        raise ValueError(f"No frequency column found.\nColumns: {cols}")
    if not ((rcol and xcol) or (zcol and tcol)):
        raise ValueError("Need either R and X columns, or |Z| and phase columns.\n"
                         f"Columns: {cols}")

    for c in (fcol, rcol, xcol, zcol, tcol):
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if rcol and xcol:
        df = df.dropna(subset=[fcol, rcol, xcol])
        make_Z = lambda d: d[rcol].to_numpy() + 1j * d[xcol].to_numpy()
    else:
        df = df.dropna(subset=[fcol, zcol, tcol])
        in_rad = "rad" in tcol.lower()

        def make_Z(d):
            th = d[tcol].to_numpy()
            th = th if in_rad else np.radians(th)
            return d[zcol].to_numpy() * np.exp(1j * th)

    id_col = find_column(cols, RUN_ID_NAMES)
    name_col = find_column(cols, RUN_NAME_NAMES)
    group_col = id_col or name_col
    stem = os.path.splitext(os.path.basename(path))[0]

    datasets = []
    if group_col is None:
        datasets.append(Dataset(stem, path, df[fcol].to_numpy(float), make_Z(df)))
    else:
        for key, d in df.groupby(group_col, sort=False):
            if name_col and name_col != group_col:
                name = str(d[name_col].iloc[0])
            else:
                name = str(key)
            datasets.append(Dataset(name, path, d[fcol].to_numpy(float), make_Z(d)))
    return datasets


# --------------------------------------------------------------------------- #
#  Plot definitions
# --------------------------------------------------------------------------- #
#  key -> (axis label, unit, kind)   kind: "mag" | "phase" | "lin"
QUANTITIES = {
    "R":     ("R = Re(Z)", "Ω", "lin"),
    "X":     ("X = Im(Z)", "Ω", "lin"),
    "Zmag":  ("|Z|", "Ω", "mag"),
    "theta": ("θ  (°)", "", "phase"),
    "G":     ("G = Re(Y)", "S", "lin"),
    "B":     ("B = Im(Y)", "S", "lin"),
    "Ymag":  ("|Y|", "S", "mag"),
    "phi":   ("φ  (°)", "", "phase"),
    "Cs":    ("Cs", "F", "lin"),
    "Rs":    ("Rs", "Ω", "lin"),
    "Cp":    ("Cp", "F", "lin"),
    "Rp":    ("Rp", "Ω", "lin"),
}

NYQUIST = "Nyquist  (-X vs R)"
PLOT_TYPES = {
    "Impedance  (R, X)":        ["R", "X"],
    "Admittance  (G, B)":       ["G", "B"],
    "Bode – Impedance (|Z|, θ)": ["Zmag", "theta"],
    "Bode – Admittance (|Y|, φ)": ["Ymag", "phi"],
    "Cs – Rs  (series)":        ["Cs", "Rs"],
    "Cp – Rp  (parallel)":      ["Cp", "Rp"],
    NYQUIST:                    None,
}


class LogMinorEng(EngFormatter):
    """Label minor log ticks only when fewer than two decade ticks are visible."""

    def __call__(self, x, pos=None):
        lo, hi = sorted(self.axis.get_view_interval())
        if lo > 0:
            n_major = np.floor(np.log10(hi)) - np.ceil(np.log10(lo)) + 1
            if n_major < 2:
                return super().__call__(x, pos)
        return ""


def set_axis_format(axis, unit, log):
    axis.set_major_formatter(EngFormatter(unit=unit))
    if log:
        axis.set_minor_formatter(LogMinorEng(unit=unit))


def robust_ylim(ax, ys, log):
    """Zoom the y axis to the 1st–99th percentile so resonance spikes don't
    squash the rest of the curve."""
    y = np.concatenate([np.asarray(v, float) for v in ys])
    y = y[np.isfinite(y) & ((y > 0) if log else True)]
    if y.size < 5:
        return
    lo, hi = np.percentile(y, [1, 99])
    if log:
        pad = (hi / lo) ** 0.08
        ax.set_ylim(lo / pad, hi * pad)
    else:
        pad = 0.08 * (hi - lo) or abs(hi) * 0.05 or 1.0
        ax.set_ylim(lo - pad, hi + pad)


def draw_plot(fig, datasets, colors, plot_type, opts):
    """Render the selected plot type into a matplotlib Figure."""
    fig.clear()
    if not datasets:
        fig.text(0.5, 0.5, "Open a CSV file and select one or more runs",
                 ha="center", va="center", fontsize=13, color="gray")
        return

    style = dict(lw=1.4, marker="o" if opts["markers"] else None, ms=3)

    # ---------------------------- Nyquist ---------------------------- #
    if plot_type == NYQUIST:
        ax = fig.add_subplot(111)
        for ds, c in zip(datasets, colors):
            ax.plot(ds.q["R"], -ds.q["X"], color=c, label=ds.name, **style)
        ax.set_xlabel("R = Re(Z)")
        ax.set_ylabel("−X = −Im(Z)")
        set_axis_format(ax.xaxis, "Ω", False)
        set_axis_format(ax.yaxis, "Ω", False)
        if opts["equal"]:
            ax.set_aspect("equal", adjustable="datalim")
        ax.grid(opts["grid"], which="both", alpha=0.35)
        ax.legend(fontsize=9)
        ax.set_title("Nyquist plot")
        fig.tight_layout()
        return

    # ------------------------ frequency plots ------------------------ #
    keys = PLOT_TYPES[plot_type]
    axes = fig.subplots(len(keys), 1, sharex=True)
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, keys):
        label, unit, kind = QUANTITIES[key]
        use_db = opts["db"] and kind == "mag"
        ys = []
        for ds, c in zip(datasets, colors):
            y = ds.q[key]
            if use_db:
                with np.errstate(divide="ignore"):
                    y = 20 * np.log10(y)
            ax.plot(ds.f, y, color=c, label=ds.name, **style)
            ys.append(y)
        log_y = kind == "mag" and not use_db or (kind == "lin" and opts["logy"])

        # y axis scaling / formatting
        if kind == "phase":
            ax.set_ylabel(label)
        elif use_db:
            ax.set_ylabel(f"{label}  (dB re 1 {unit})")
        else:
            if log_y:
                ax.set_yscale("log", nonpositive="mask")
            set_axis_format(ax.yaxis, unit, log_y)
            ax.set_ylabel(label)
        if opts["robust"]:
            robust_ylim(ax, ys, log_y)

        ax.grid(opts["grid"], which="major", alpha=0.45)
        ax.grid(opts["grid"], which="minor", alpha=0.15)

    bottom = axes[-1]
    if opts["logx"]:
        bottom.set_xscale("log")
    set_axis_format(bottom.xaxis, "Hz", opts["logx"])
    bottom.set_xlabel("Frequency")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].set_title(plot_type.replace("  ", " "))
    fig.tight_layout()


# --------------------------------------------------------------------------- #
#  GUI
# --------------------------------------------------------------------------- #
class ImpedancePlotterApp(tk.Tk):
    def __init__(self, files=()):
        super().__init__()
        self.title("Impedance Plotter")
        self.geometry("1280x800")
        self.minsize(900, 600)

        self.datasets = []                      # all loaded runs
        self.palette = matplotlib.colormaps["tab10"].colors

        self.plot_type = tk.StringVar(value=next(iter(PLOT_TYPES)))
        self.opt = {name: tk.BooleanVar(value=val) for name, val in
                    dict(logx=True, logy=False, db=False, robust=False,
                         grid=True, markers=False, equal=False).items()}

        self._build_ui()
        for f in files:
            self._load_file(f)
        self.redraw()

    # ---------------- layout ---------------- #
    def _build_ui(self):
        self.status = tk.StringVar(value="No data loaded.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=(6, 2)).pack(side=tk.BOTTOM, fill=tk.X)

        side = ttk.Frame(self, padding=8)
        side.pack(side=tk.LEFT, fill=tk.Y)

        # file buttons
        fbox = ttk.LabelFrame(side, text="Data", padding=6)
        fbox.pack(fill=tk.X)
        ttk.Button(fbox, text="Open CSV…", command=self.open_files).pack(fill=tk.X)
        row = ttk.Frame(fbox)
        row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(row, text="Remove", command=self.remove_selected).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(row, text="Clear all", command=self.clear_all).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        # run list
        rbox = ttk.LabelFrame(side, text="Runs (Ctrl/Shift-click for several)", padding=6)
        rbox.pack(fill=tk.BOTH, expand=True, pady=6)
        self.runlist = tk.Listbox(rbox, selectmode=tk.EXTENDED, width=34,
                                  exportselection=False, activestyle="none")
        sb = ttk.Scrollbar(rbox, orient=tk.VERTICAL, command=self.runlist.yview)
        self.runlist.configure(yscrollcommand=sb.set)
        self.runlist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.runlist.bind("<<ListboxSelect>>", lambda e: self.redraw())

        # plot type
        pbox = ttk.LabelFrame(side, text="Plot", padding=6)
        pbox.pack(fill=tk.X)
        for name in PLOT_TYPES:
            ttk.Radiobutton(pbox, text=name, value=name, variable=self.plot_type,
                            command=self.redraw).pack(anchor=tk.W)

        # options
        obox = ttk.LabelFrame(side, text="Options", padding=6)
        obox.pack(fill=tk.X, pady=6)
        labels = dict(logx="Log frequency axis",
                      logy="Log Y for R / X / G / B / C axes",
                      db="Bode magnitude in dB",
                      robust="Ignore spikes in Y range (1–99 %)",
                      grid="Grid",
                      markers="Show markers",
                      equal="Nyquist: equal aspect")
        for key, text in labels.items():
            ttk.Checkbutton(obox, text=text, variable=self.opt[key],
                            command=self.redraw).pack(anchor=tk.W)

        ttk.Button(side, text="Export computed data (CSV)…",
                   command=self.export_csv).pack(fill=tk.X)

        # plot area
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    # ---------------- data handling ---------------- #
    def open_files(self):
        paths = filedialog.askopenfilenames(
            title="Open impedance CSV",
            filetypes=[("CSV files", "*.csv *.txt"), ("All files", "*.*")])
        for p in paths:
            self._load_file(p)
        self.redraw()

    def _load_file(self, path):
        try:
            new = load_datasets(path)
        except Exception as exc:  # show any parse problem to the user
            messagebox.showerror("Could not load file",
                                 f"{os.path.basename(path)}\n\n{exc}")
            return
        existing = {d.name for d in self.datasets}
        stem = os.path.splitext(os.path.basename(path))[0]
        for ds in new:
            if ds.name in existing:       # disambiguate duplicate run names
                ds.name = f"{ds.name} ({stem})"
            existing.add(ds.name)
            self.datasets.append(ds)
            idx = len(self.datasets) - 1
            self.runlist.insert(tk.END, f"{ds.name}   [{len(ds.f)} pts]")
            self.runlist.itemconfig(idx, foreground=self._color(idx))
            self.runlist.selection_set(idx)

    def _color(self, idx):
        return matplotlib.colors.to_hex(self.palette[idx % len(self.palette)])

    def remove_selected(self):
        for idx in sorted(self.runlist.curselection(), reverse=True):
            self.runlist.delete(idx)
            del self.datasets[idx]
        for i in range(len(self.datasets)):
            self.runlist.itemconfig(i, foreground=self._color(i))
        self.redraw()

    def clear_all(self):
        self.runlist.delete(0, tk.END)
        self.datasets.clear()
        self.redraw()

    def selected(self):
        idxs = self.runlist.curselection()
        return [self.datasets[i] for i in idxs], [self._color(i) for i in idxs]

    # ---------------- plotting / export ---------------- #
    def redraw(self):
        sel, colors = self.selected()
        opts = {k: v.get() for k, v in self.opt.items()}
        try:
            draw_plot(self.fig, sel, colors, self.plot_type.get(), opts)
        except Exception as exc:
            messagebox.showerror("Plot error", str(exc))
        self.canvas.draw_idle()

        if not self.datasets:
            self.status.set("No data loaded.  Use 'Open CSV…' to add a file.")
        elif sel:
            fmin = min(d.f.min() for d in sel)
            fmax = max(d.f.max() for d in sel)
            fmt = EngFormatter(unit="Hz")
            self.status.set(f"{len(sel)} of {len(self.datasets)} run(s) shown  |  "
                            f"{sum(len(d.f) for d in sel)} points  |  "
                            f"{fmt(fmin)} – {fmt(fmax)}")
        else:
            self.status.set("Select one or more runs in the list.")

    def export_csv(self):
        sel, _ = self.selected()
        if not sel:
            messagebox.showinfo("Export", "Select at least one run first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export computed data", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if path:
            pd.concat([d.to_frame() for d in sel]).to_csv(path, index=False)
            self.status.set(f"Exported {len(sel)} run(s) to {path}")


if __name__ == "__main__":
    app = ImpedancePlotterApp(files=sys.argv[1:])
    app.mainloop()
