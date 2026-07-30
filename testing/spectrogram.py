#!/usr/bin/env python3
"""
PicoScope 2-Channel Spectrogram Plotter
========================================

IMPORTANT: PicoScope's native ".psdata" file is a closed, proprietary binary
format. Pico Technology has confirmed on their own support forum that it
cannot be parsed by third-party (including Python) code - there is no public
spec or library for it.

The supported workflow is to export/convert your capture to CSV or MATLAB
(.mat) format first, then read *that* file in Python. You can do this either
from the PicoScope 6/7 GUI or from the command line:

    GUI:  File -> Save As... -> choose "CSV" or "MATLAB (*.mat)"

    CLI (Windows, PicoScope installed):
        picoscope /c "your_capture.psdata" /f mat /q /b 1
        picoscope /c "your_capture.psdata" /f csv /q /b 1

This script then loads the resulting .csv or .mat file (2 channels) and
plots a spectrogram for each channel.

Usage
-----
    pip install numpy scipy matplotlib pandas

    python psdata_spectrogram.py capture.csv
    python psdata_spectrogram.py capture.mat --nperseg 2048
    python psdata_spectrogram.py capture.csv --fs 1e6 --out spectrogram.png

Notes
-----
- For CSV: the script auto-detects the PicoScope export header (it usually
  has a name row and a units row before the numeric data) and auto-detects
  the time units (s / ms / us / ns) to compute the sample rate.
- For MAT: it looks for the standard PicoScope variables 'Tinterval' plus
  two channel arrays (commonly named 'A' and 'B', or 'Channel A'/'Channel B').
- Use --fs to override the auto-detected sample rate if needed.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def load_csv(path, delimiter=","):
    """Load a PicoScope CSV export: time column + 2 channel columns."""
    import pandas as pd

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [f.readline() for _ in range(5)]

    def is_numeric_row(line):
        parts = [p.strip().strip('"') for p in line.strip().split(delimiter) if p.strip() != ""]
        if not parts:
            return False
        try:
            [float(p) for p in parts]
            return True
        except ValueError:
            return False

    # Figure out how many header rows to skip (name row, units row, etc.)
    skip = 0
    for line in raw_lines:
        if not line:
            break
        if is_numeric_row(line):
            break
        skip += 1

    header_cols = [p.strip().strip('"') for p in raw_lines[0].split(delimiter)]

    df = pd.read_csv(path, delimiter=delimiter, skiprows=skip, header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns (time + 2 channels), found {df.shape[1]}. "
            "Check --delimiter or the file contents."
        )

    time_col = df.iloc[:, 0].to_numpy(dtype=float)
    ch_a = df.iloc[:, 1].to_numpy(dtype=float)
    ch_b = df.iloc[:, 2].to_numpy(dtype=float)

    # Detect time units from ALL header rows (name row + units row, e.g.
    # PicoScope often puts "(ms)" on a separate row from the column names)
    header_text = " ".join(raw_lines[:skip]).lower()
    scale = 1.0
    if "(ms)" in header_text or "millisecond" in header_text:
        scale = 1e-3
    elif "(us)" in header_text or "\u00b5s" in header_text or "microsecond" in header_text:
        scale = 1e-6
    elif "(ns)" in header_text or "nanosecond" in header_text:
        scale = 1e-9

    time_seconds = time_col * scale
    fs = 1.0 / np.mean(np.diff(time_seconds))

    name_a = header_cols[1] if len(header_cols) > 1 and header_cols[1] else "Channel A"
    name_b = header_cols[2] if len(header_cols) > 2 and header_cols[2] else "Channel B"
    return fs, ch_a, ch_b, name_a, name_b


def load_mat(path):
    """Load a PicoScope MATLAB export (Tinterval + 2 channel arrays)."""
    from scipy.io import loadmat

    mat = loadmat(path)
    if "Tinterval" not in mat:
        raise ValueError("Could not find 'Tinterval' in the .mat file - unexpected format.")

    fs = 1.0 / float(np.squeeze(mat["Tinterval"]))

    skip_keys = {"Tstart", "Tinterval", "Length", "ExtraSamples", "RequestedLength"}
    channel_keys = [
        k for k in mat.keys() if not k.startswith("__") and k not in skip_keys
    ]
    if len(channel_keys) < 2:
        raise ValueError(f"Expected 2 channel arrays in the .mat file, found: {channel_keys}")

    ch_a = np.squeeze(mat[channel_keys[0]]).astype(float)
    ch_b = np.squeeze(mat[channel_keys[1]]).astype(float)
    return fs, ch_a, ch_b, channel_keys[0], channel_keys[1]


#def plot_spectrogram1(ch_a, ch_b, fs, name_a, name_b, nperseg=1024, noverlap=None, out=None, show=False):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    name_a="recieved signal"
    name_b="transmitted signal"
    for ax, data, name in zip(axes, (ch_b, ch_a), (name_b, name_a)):
        nps = min(nperseg, len(data))
        f, t, sxx = signal.spectrogram(data, fs=fs, nperseg=nps, noverlap=noverlap)
        power_db = 10 * np.log10(sxx + 1e-20)
        pcm = ax.pcolormesh(t, f, power_db, shading="gouraud", cmap="jet")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(name)
        fig.colorbar(pcm, ax=ax, label="Power (dB)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"noverlap={noverlap}")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved: {os.path.abspath(out)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_spectrogram1(ch_a, ch_b, fs, name_a, name_b, nperseg=1024, noverlap=None, out=None, show=False):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    name_a="Recieved signal"
    name_b="Transmitted signal"
    # --- compute both spectrograms first so we can share one color scale ---
    specs = []
    for data in (ch_b, ch_a):
        nps = min(nperseg, len(data))
        f, t, sxx = signal.spectrogram(data, fs=fs, nperseg=nps, noverlap=noverlap)
        power_db = 10 * np.log10(sxx + 1e-20)
        specs.append((f, t, power_db))

    vmin = min(p.min() for _, _, p in specs)
    vmax = max(p.max() for _, _, p in specs)

    for ax, (f, t, power_db), name in zip(axes, specs, (name_b, name_a)):
        pcm = ax.pcolormesh(t, f, power_db, shading="gouraud", cmap="jet",
                             vmin=vmin, vmax=vmax)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(name)
        fig.colorbar(pcm, ax=ax, label="Power (dB)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"noverlap={noverlap}")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved: {os.path.abspath(out)}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_spectrogram2(ch_a, ch_b, fs, name_a, name_b, nperseg=1024, noverlap=None, out=None, show=False):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    name_a="Recieved signal"
    name_b="Transmitted signal"
    # --- compute both spectrograms first so we can share one color scale ---
    specs = []
    for data in (ch_b, ch_a):
        nps = min(nperseg, len(data))
        f, t, sxx = signal.spectrogram(data, fs=fs, nperseg=nps, noverlap=noverlap)
        power_db = 10 * np.log10(sxx + 1e-20)
        specs.append((f, t, power_db))

    vmin = min(p.min() for _, _, p in specs)
    vmax = max(p.max() for _, _, p in specs)

    for ax, (f, t, power_db), name in zip(axes, specs, (name_b, name_a)):
        pcm = ax.pcolormesh(t, f, power_db, shading="gouraud", cmap="jet",
                             vmin=vmin, vmax=vmax)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(100e3,400e3)
        ax.set_title(name)
        fig.colorbar(pcm, ax=ax, label="Power (dB)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"noverlap={noverlap}")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved: {os.path.abspath(out)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


'''def plot_spectrogram2(ch_a, ch_b, fs, name_a, name_b, nperseg=1024, noverlap=None, out=None, show=False):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    name_a="recieved signal"
    name_b="transmitted signal"
    for ax, data, name in zip(axes, (ch_b, ch_a), (name_b, name_a)):
        nps = min(nperseg, len(data))
        f, t, sxx = signal.spectrogram(data, fs=fs, nperseg=nps, noverlap=noverlap)
        power_db = 10 * np.log10(sxx + 1e-20)
        pcm = ax.pcolormesh(t, f, power_db, shading="gouraud", cmap="jet")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(100e3,400e3)
        ax.set_title(name)
        fig.colorbar(pcm, ax=ax, label="Power (dB)")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"noverlap={noverlap}")
    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150)
        print(f"Saved: {os.path.abspath(out)}")

    if show:
        plt.show()
    else:
        plt.close(fig)
'''

def main():
    parser = argparse.ArgumentParser(
        description="Plot spectrograms from a 2-channel PicoScope capture (CSV or MAT export)."
    )
    parser.add_argument("file", help="Path to the exported .csv or .mat file (NOT the raw .psdata)")
    parser.add_argument("--fs", type=float, default=None, help="Override sample rate in Hz")
    parser.add_argument("--nperseg", type=int, default=1024, help="FFT window length (default 1024)")
    parser.add_argument(
        "--noverlap",
        type=int,
        default=None,
        help="Overlap (samples) for the FIRST plot (_001). Default: scipy's default (nperseg // 8).",
    )
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Folder to save the PNGs in. Default: same folder as the input file.",
    )
    parser.add_argument("--show", action="store_true", help="Also display the plots on screen after saving")
    args = parser.parse_args()

    ext = args.file.lower().rsplit(".", 1)[-1] if "." in args.file else ""

    if ext == "csv":
        fs, ch_a, ch_b, name_a, name_b = load_csv(args.file, args.delimiter)
    elif ext == "mat":
        fs, ch_a, ch_b, name_a, name_b = load_mat(args.file)
    elif ext == "psdata":
        print(
            "ERROR: '.psdata' is a proprietary PicoScope format and cannot be read directly.\n"
            "Export it first from PicoScope 6/7:\n"
            "  GUI:  File > Save As... > choose CSV or MATLAB (*.mat)\n"
            '  CLI:  picoscope /c "your_capture.psdata" /f mat /q /b 1\n'
            "Then run this script on the resulting .csv or .mat file."
        )
        sys.exit(1)
    else:
        print(f"Unsupported file extension '.{ext}'. Please provide a .csv or .mat file.")
        sys.exit(1)

    if args.fs is not None:
        fs = args.fs

    print(f"Sample rate: {fs:.4g} Hz | Samples: {len(ch_a)} | Duration: {len(ch_a) / fs:.4f} s")

    # Figure out where to save: --outdir if given, otherwise the same folder
    # the input file lives in.
    outdir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.file))
    os.makedirs(outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.file))[0]
    #out1 = os.path.join(outdir, f"{base}_noverlap.png")
    #out2 = os.path.join(outdir, f"{base}_noverlap_lim.png")
    out3 = os.path.join(outdir, f"{base}_nperseg2.png")
    out4 = os.path.join(outdir, f"{base}_nperseg2_lim.png")

    # Plot 1: noverlap as given on the command line (or scipy's default if not given)
    '''plot_spectrogram1(
        ch_a, ch_b, fs, name_a, name_b,
        nperseg=args.nperseg, noverlap=args.noverlap, out=out1, show=args.show,
    )

    #plot_spectrogram2(
        ch_a, ch_b, fs, name_a, name_b,
        nperseg=args.nperseg, noverlap=args.noverlap, out=out2, show=args.show,
    )'''

    # Plot 2: noverlap fixed at nperseg / 2
    noverlap_half = args.nperseg / 2
    plot_spectrogram1(
        ch_a, ch_b, fs, name_a, name_b,
        nperseg=args.nperseg, noverlap=noverlap_half, out=out3, show=args.show,
    )

    plot_spectrogram2(
            ch_a, ch_b, fs, name_a, name_b,
            nperseg=args.nperseg, noverlap=noverlap_half, out=out4, show=args.show,
        )


if __name__ == "__main__":
    main()
