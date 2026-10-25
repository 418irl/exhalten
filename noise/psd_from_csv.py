import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, sosfiltfilt
from scipy.integrate import trapezoid

# ---------------- Settings ----------------
CSV_PATHS = [
    "23102026_nidaq_self.csv",
    "23102026_probe_sr560_1x.csv",
    "23102026_sr560_self_1x.csv",
    "23102026_resistor_10M_1x.csv"
]
LABELS = None            # e.g. ["Amp off", "Amp on"]; None = file names
CHANNEL = None           # scope-style CSVs only; None = first data column
GAIN = [1, 1, 1, 1]  # one number for all files, or one per file
#GAIN = 1
SAMPLE_RATE = 2_000_000  # Hz; used for raw DAQ files (no time column)
NPERSEG = 8192
OVERLAP = 0.5
WINDOW = "hann"
FMIN = None              # Hz; None = auto
FMAX = None
PLOT_DB = True           # True -> dB scale (overrides linear PSD/ASD display)
PLOT_ASD = False         # linear mode: True -> V/sqrt(Hz), False -> V^2/Hz
                         # dB mode: True -> label dBV/sqrt(Hz), False -> dB re 1 V^2/Hz (same numbers)
REMOVE_DC = False
BANDLIMIT = False
BAND_LOW = 42e3          # Hz
BAND_HIGH = 52e3         # Hz
FILTER_ORDER = 4
MAX_FILES = 10
RES = 10e6               # ohms
TEMP = 23                # °C
kB = 1.380649e-23        # J/K
# ------------------------------------------

TIME_UNITS = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
VOLT_UNITS = {"nV": 1e-9, "uV": 1e-6, "µV": 1e-6, "mV": 1e-3, "V": 1.0}


def is_raw_daq_file(path):
    """True if the file is one column of numbers with no header
    (the format written by the acquisition script's np.savetxt)."""
    with open(path) as fh:
        first = fh.readline().strip()
    if first.lower().startswith("freq"):
        raise SystemExit(f"{path} is a PSD file (freq_Hz,...). "
                         f"Point CSV_PATHS at the raw data file instead.")
    fields = [x for x in first.replace(";", ",").split(",") if x.strip()]
    if len(fields) != 1:
        return False
    try:
        float(fields[0])
        return True
    except ValueError:
        return False


def load_csv(path, channel=None):
    """Return (t_seconds, v_volts, channel_name).

    Supports two formats:
      1. Raw DAQ files from the acquisition script: one column of volts,
         no header, no time column. Time is rebuilt from SAMPLE_RATE.
      2. Scope-style CSVs: header row, optional units row, time in the first column.
    """
    if is_raw_daq_file(path):
        v = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)
        t = np.arange(len(v)) / SAMPLE_RATE
        return t, v, "AI0"

    df = pd.read_csv(path, sep=None, engine="python", skip_blank_lines=True)
    df.columns = [c.strip() for c in df.columns]

    # second row normally holds units like "(ms)" and "(mV)"
    units = {}
    first = df.iloc[0]
    if pd.to_numeric(first, errors="coerce").isna().any():
        for col in df.columns:
            units[col] = str(first[col]).strip().strip("()")
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

    time_col = df.columns[0]
    if channel is None:
        channel = df.columns[1]
    if channel not in df.columns:
        raise SystemExit(f"{path}: channel '{channel}' not found. Columns: {list(df.columns)}")

    t = df[time_col].to_numpy() * TIME_UNITS.get(units.get(time_col, "s"), 1.0)
    v = df[channel].to_numpy() * VOLT_UNITS.get(units.get(channel, "V"), 1.0)
    return t, v, channel


def per_file(value, n, name):
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise SystemExit(f"{name} has {len(value)} entries but there are {n} CSV files.")
        return list(value)
    return [value] * n


def band_edges(fs):
    """Band-pass edges clipped below Nyquist; None if the band doesn't fit."""
    nyq = fs / 2
    low, high = BAND_LOW, min(BAND_HIGH, 0.99 * nyq)
    if high < BAND_HIGH:
        print(f"Note: BAND_HIGH {BAND_HIGH:.4g} Hz is above Nyquist ({nyq:.4g} Hz); using {high:.4g} Hz")
    if low <= 0 or low >= high:
        print(f"Warning: band does not fit at fs = {fs:.4g} Hz; no band-pass applied")
        return None
    return (low, high)


def compute_psd(path, gain, channel):
    t, v, ch = load_csv(path, channel)
    v = v / gain
    fs = 1.0 / np.median(np.diff(t))
    print(f"--- {path}  [{ch}]  gain = {gain:g}")

    if REMOVE_DC:
        dc = np.mean(v)
        v = v - dc
        print(f"DC offset removed: {dc:.6g} V")

    band = None
    if BANDLIMIT:
        band = band_edges(fs)
        if band is not None:
            sos = butter(FILTER_ORDER, band, btype="bandpass", fs=fs, output="sos")
            v = sosfiltfilt(sos, v)
            print(f"Band-pass: {band[0]:.4g} – {band[1]:.4g} Hz")

    nperseg = min(NPERSEG, len(v))
    noverlap = int(nperseg * OVERLAP)
    f, Pxx = welch(v, fs=fs, window=WINDOW, nperseg=nperseg,
                   noverlap=noverlap, detrend="constant", scaling="density")

    print(f"Samples: {len(v)}  fs: {fs:.6g} Hz  duration: {t[-1]-t[0]:.6g} s")
    print(f"nperseg: {nperseg}  resolution: {fs/nperseg:.4g} Hz")
    print(f"RMS (AC): {np.std(v):.6g} V")
    if band is not None:
        m = (f >= band[0]) & (f <= band[1])
        print(f"RMS in band (from PSD): {np.sqrt(trapezoid(Pxx[m], f[m])):.6g} V")

    return {"f": f, "Pxx": Pxx, "fs": fs, "nperseg": nperseg, "channel": ch, "band": band}


def add_horizontal_cursor(fig, ax):
    """Click/drag in the plot to move a horizontal cursor. 'c' hides/shows it.
    Ignored while the zoom/pan tool is active."""
    line = ax.axhline(np.nan, color="red", lw=1.2, ls="--")
    label = ax.text(0.01, np.nan, "", color="red", transform=ax.get_yaxis_transform(),
                    va="bottom", ha="left",
                    bbox=dict(fc="white", ec="red", alpha=0.85, pad=2))
    state = {"dragging": False}

    def update(y):
        if y is None:
            return
        if PLOT_DB:
            psd = 10 ** (y / 10)
            txt = f"{y:.2f} dB   ({np.sqrt(psd):.4g} V/√Hz)"
        else:
            if y <= 0:
                return
            if PLOT_ASD:
                txt = f"{y:.4g} V/√Hz   ({y**2:.4g} V²/Hz)"
            else:
                txt = f"{y:.4g} V²/Hz   ({np.sqrt(y):.4g} V/√Hz)"
        line.set_ydata([y, y])
        label.set_position((0.01, y))
        label.set_text(txt)
        fig.canvas.draw_idle()

    def tool_active():
        tb = getattr(fig.canvas, "toolbar", None)
        return bool(tb and getattr(tb, "mode", ""))

    def on_legend(e):
        leg = ax.get_legend()
        return leg is not None and leg.contains(e)[0]

    def on_press(e):
        if e.inaxes is ax and e.button == 1 and not tool_active() and not on_legend(e):
            state["dragging"] = True
            update(e.ydata)

    def on_move(e):
        if state["dragging"] and e.inaxes is ax:
            update(e.ydata)

    def on_release(e):
        state["dragging"] = False

    def on_key(e):
        if e.key == "c":
            vis = not line.get_visible()
            line.set_visible(vis)
            label.set_visible(vis)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)


def add_vertical_cursors(fig, ax):
    """Two vertical cursors for bandwidth. Right-click/drag moves the nearest one.
    'v' hides/shows them. Ignored while the zoom/pan tool is active."""
    x0, x1 = ax.get_xlim()
    pos = [x0 * (x1 / x0) ** 0.25, x0 * (x1 / x0) ** 0.75]   # start at 1/4 and 3/4 (log scale)
    lines = [ax.axvline(p, color=c, lw=1.2, ls="--") for p, c in zip(pos, ("green", "purple"))]
    label = ax.text(0.99, 0.98, "", transform=ax.transAxes, va="top", ha="right",
                    bbox=dict(fc="white", ec="gray", alpha=0.85, pad=2))
    state = {"idx": None}

    def refresh():
        lo, hi = sorted(pos)
        label.set_text(f"f1 = {pos[0]:.5g} Hz\nf2 = {pos[1]:.5g} Hz\n"
                       f"BW = {hi - lo:.5g} Hz\nf2/f1 = {hi / lo:.3g}")
        for ln, p in zip(lines, pos):
            ln.set_xdata([p, p])
        fig.canvas.draw_idle()

    def tool_active():
        tb = getattr(fig.canvas, "toolbar", None)
        return bool(tb and getattr(tb, "mode", ""))

    def move(x):
        if x is not None and x > 0:
            pos[state["idx"]] = x
            refresh()

    def on_press(e):
        if e.inaxes is ax and e.button == 3 and not tool_active() and e.xdata and e.xdata > 0:
            state["idx"] = int(np.argmin([abs(np.log(p / e.xdata)) for p in pos]))
            move(e.xdata)

    def on_move(e):
        if state["idx"] is not None and e.inaxes is ax:
            move(e.xdata)

    def on_release(e):
        state["idx"] = None

    def on_key(e):
        if e.key == "v":
            vis = not lines[0].get_visible()
            for ln in lines:
                ln.set_visible(vis)
            label.set_visible(vis)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()


def add_legend_toggle(fig, ax, leg):
    """Click a legend entry to hide/show that curve."""
    curves = [l for l in ax.get_lines() if not l.get_label().startswith("_")]
    lookup = {}
    for entry, text, curve in zip(leg.get_lines(), leg.get_texts(), curves):
        entry.set_picker(5)
        text.set_picker(True)
        lookup[entry] = lookup[text] = (entry, text, curve)

    def on_pick(e):
        if e.artist not in lookup:
            return
        entry, text, curve = lookup[e.artist]
        vis = not curve.get_visible()
        curve.set_visible(vis)
        entry.set_alpha(1.0 if vis else 0.2)
        text.set_alpha(1.0 if vis else 0.4)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)


def main():
    n = len(CSV_PATHS)
    if not 1 <= n <= MAX_FILES:
        raise SystemExit(f"Put between 1 and {MAX_FILES} files in CSV_PATHS (got {n}).")

    # theoretical thermal noise of RES
    thermal_asd = np.sqrt(4 * kB * (TEMP + 273.15) * RES)    # V/√Hz
    thermal_psd = thermal_asd ** 2                           # V²/Hz
    print(f"Resistor: {RES:g} Ω")
    print(f"Thermal noise: {thermal_asd:.4g} V/√Hz = {10*np.log10(thermal_psd):.2f} dB re 1 V²/Hz")
    print(f"Thermal noise RMS over band: {thermal_asd*np.sqrt(BAND_HIGH-BAND_LOW):.4g} V")

    gains = per_file(GAIN, n, "GAIN")
    labels = LABELS if LABELS is not None else [os.path.splitext(os.path.basename(p))[0]
                                                 for p in CSV_PATHS]
    labels = per_file(labels, n, "LABELS")
    results = [compute_psd(p, g, CHANNEL) for p, g in zip(CSV_PATHS, gains)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for r, lab in zip(results, labels):
        f, P = r["f"], r["Pxx"]
        # skip DC and the Nyquist bin (scipy doesn't double it, so it sits ~3 dB low)
        keep = (f > 0) & (f < f[-1])
        if r["band"] is not None:
            keep &= (f >= r["band"][0]) & (f <= r["band"][1])
        if PLOT_DB:
            ax.semilogx(f[keep], 10 * np.log10(P[keep]), lw=1, alpha=0.85, label=lab)
        else:
            y = np.sqrt(P[keep]) if PLOT_ASD else P[keep]
            ax.loglog(f[keep], y, lw=1, alpha=0.85, label=lab)

    # theory line
    if PLOT_DB:
        y_theory = 10 * np.log10(thermal_psd)
    else:
        y_theory = thermal_asd if PLOT_ASD else thermal_psd
    ax.axhline(y_theory, color="k", ls="--", lw=1,
               label=f"4kTR, {RES:g} Ω ({thermal_asd*1e9:.3g} nV/√Hz)")

    if PLOT_DB:
        ax.set_ylabel("ASD (dBV/√Hz)" if PLOT_ASD else "PSD (dB re 1 V²/Hz)")
    else:
        ax.set_ylabel("ASD (V/√Hz)" if PLOT_ASD else "PSD (V²/Hz)")
    ax.set_xlabel("Frequency (Hz)")
    if n == 1:
        r = results[0]
        ax.set_title(f"Welch PSD – {r['channel']}  (fs = {r['fs']:.4g} Hz, nperseg = {r['nperseg']})")
    else:
        ax.set_title(f"Welch PSD – {n} files  (nperseg = {NPERSEG})")
    ax.grid(True, which="both", alpha=0.3)
    leg = ax.legend(loc="best", fontsize="small")
    add_legend_toggle(fig, ax, leg)

    if FMIN or FMAX:
        f_lo = min(r["f"][1] for r in results)
        f_hi = max(r["f"][-1] for r in results)
        ax.set_xlim(FMIN or f_lo, FMAX or f_hi)

    add_horizontal_cursor(fig, ax)
    add_vertical_cursors(fig, ax)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
