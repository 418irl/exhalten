import os
import sys
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, hilbert


# ============================================================
# PARAMETERS
# ============================================================

ADC_BITS = 12
ADC_MAX = 2**ADC_BITS - 1
ADC_VREF = 3.3

FS = 1.1e6          # 1.1 MHz sampling rate
FC = 110e3          # 110 kHz signal

TEMPLATE_LENGTH = 260
SIGNAL_VPP = 1.0

MASK_SAMPLES = 1500

SPEED_OF_SOUND = 1500.0  # m/s


# ============================================================
# 0. COMMAND-LINE ARGS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched-filter ADC processing. "
                     "Either point it at an existing data file with -f, "
                     "or leave -f out to paste/type samples straight into the terminal."
    )
    parser.add_argument(
        "file", nargs="?", default=None,
        help="Path to an existing ADC data file (one sample per line, or whitespace-separated). "
             "Leave this out to paste/type samples straight into the terminal instead."
    )
    parser.add_argument(
        "-g", "--gain", type=str, default="1x",
        help="Gain label used when auto-naming a newly entered/saved file (default: 1x)"
    )
    parser.add_argument(
        "-d", "--distance", type=str, default=None,
        help="Distance/label used when auto-naming a newly entered/saved file, e.g. 23m"
    )
    parser.add_argument(
        "-n", "--name", type=str, default=None,
        help="Extra label to prefix the auto-generated filename, e.g. sideReflex"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default=".",
        help="Directory to save newly entered data into (default: current directory)"
    )
    return parser.parse_args()


# ============================================================
# 0b. READ ADC DATA PASTED/TYPED INTO THE TERMINAL
# ============================================================

def read_adc_data_from_terminal():
    """
    Reads whitespace/newline-separated numeric samples typed or pasted
    into the terminal. Works both when a person is typing interactively
    (finish with an empty line or Ctrl+D) and when data is piped in
    (e.g. `python matched_filter_dvl.py < some_dump.txt`).
    """
    if sys.stdin.isatty():
        print("Paste or type ADC sample values (whitespace or newline separated).")
        print("Press Enter on an empty line, or Ctrl+D, when finished.\n")

    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
    except EOFError:
        pass

    values = " ".join(lines).split()

    if not values:
        print("No data was entered.")
        sys.exit(1)

    try:
        data = np.array([float(v) for v in values])
    except ValueError as e:
        print(f"Could not parse one of the entered values as a number: {e}")
        sys.exit(1)

    return data


# ============================================================
# 0c. AUTO-SAVE ENTERED DATA TO A FILE
# ============================================================

def save_adc_data(data, gain, distance, name, outdir):
    """
    Saves newly entered samples to a text file, one value per line,
    auto-naming it in the same style as the recorded capture files
    (e.g. 1xGain_23m_50kSamples.txt / sideReflex_1xGain_50kSamples.txt).
    """
    n_samples = len(data)
    sample_label = f"{n_samples // 1000}kSamples" if n_samples >= 1000 else f"{n_samples}Samples"
    distance_label = distance if distance else "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [p for p in [name, f"{gain}Gain", distance_label, sample_label, timestamp] if p]
    filename = "_".join(parts) + ".txt"

    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)

    np.savetxt(filepath, data, fmt="%.6f")
    print(f"Saved {n_samples} samples to: {filepath}")

    return filepath


def prompt_for_labels(args):
    gain = args.gain
    distance = args.distance
    if distance is None and sys.stdin.isatty():
        distance = input("Distance/label for filename (e.g. 23m) [unknown]: ").strip() or None
    return gain, distance


# ============================================================
# 1. READ RAW ADC DATA
# ============================================================

args = parse_args()

if args.file:
    filename = args.file
    adc_data = np.loadtxt(filename)
else:
    gain, distance = prompt_for_labels(args)
    adc_data = read_adc_data_from_terminal()
    filename = save_adc_data(adc_data, gain, distance, args.name, args.outdir)

print(f"Number of ADC samples: {len(adc_data)}")

# Plots get saved next to the data file
plot_dir = os.path.dirname(filename) or "."
plot_base = os.path.splitext(os.path.basename(filename))[0]


# ============================================================
# 2. CONVERT ADC COUNTS TO VOLTAGE
# ============================================================

voltage = (adc_data / ADC_MAX) * ADC_VREF

# Remove midpoint offset (approximately 1.65 V)
signal = voltage - ADC_VREF / 2

# Remove any remaining DC component
signal = signal - np.mean(signal)


# ============================================================
# 3. CREATE 260-SAMPLE 110 kHz REFERENCE
# ============================================================

t_template = np.arange(TEMPLATE_LENGTH) / FS

#amplitude = SIGNAL_VPP / 2       # 1 Vpp -> 0.5 V peak

template = np.sin(
    2 * np.pi * FC * t_template
)


# ============================================================
# 4. NORMALIZE TEMPLATE ENERGY
# ============================================================
'''
energy = np.sum(template ** 2)

template = template / np.sqrt(energy)

normalized_energy = np.sum(template ** 2)

print("\nTemplate:")
print(f"Length:             {len(template)} samples")
print(f"Original energy:    {energy:.6f}")
print(f"Normalized energy:  {normalized_energy:.6f}")'''


# ============================================================
# 5. MATCHED FILTER / CORRELATION
# ============================================================

matched_filter = correlate(
    signal,
    template,
    mode="valid"
)
energy = np.sum(template ** 2)
matched_filter=matched_filter/energy

# ============================================================
# 6. MASK FIRST 1500 SAMPLES
# ============================================================

# Ignore the first 1500 matched-filter samples
masked_output = matched_filter.copy()
masked_output[:MASK_SAMPLES] = 0
#envelope= np.abs(hilbert(masked_output[:30000]))
envelope= np.abs(hilbert(masked_output))
# ============================================================
# 7. FIND BIGGEST PEAK
# ============================================================

peak_index = np.argmax(np.abs(envelope))

peak_value = envelope[peak_index]

peak_magnitude = np.abs(peak_value)


# ============================================================
# 8. CONVERT PEAK LOCATION TO TIME
# ============================================================

peak_time = peak_index / FS
distance= (peak_time)*750 #1500/2

print("\nMatched Filter Result")
print("---------------------")
print(f"Mask length:        {MASK_SAMPLES} samples")
print(f"Peak sample:        {peak_index}")
print(f"Peak correlation:   {peak_value:.6f}")
print(f"Peak magnitude:     {peak_magnitude:.6f}")
print(f"Peak time:          {peak_time * 1e6:.3f} us")
print("distance:",distance)

# ============================================================
# 9. PLOT RAW SIGNAL
# ============================================================

signal_samples = np.arange(len(signal))

fig_signal, ax_signal = plt.subplots(figsize=(12, 6))

# ------------------------------------------------------------
# Bottom axis = Sample Number
# ------------------------------------------------------------

ax_signal.plot(signal_samples, signal)

ax_signal.set_xlabel("Sample Number")
ax_signal.set_ylabel("Voltage (V)")
ax_signal.set_title("Received ADC Signal")

ax_signal.set_xlim(0, len(signal) - 1)

ax_signal.grid(True)


# ------------------------------------------------------------
# Top axis #1 = Time
# ------------------------------------------------------------

ax_signal_time = ax_signal.twiny()

ax_signal_time.set_xlim(ax_signal.get_xlim())

ax_signal_time.set_xticks(ax_signal.get_xticks())

ax_signal_time.set_xticklabels(
    [
        f"{(x / FS) * 1000:.1f}"
        for x in ax_signal.get_xticks()
    ]
)

ax_signal_time.set_xlabel(
    "Time (ms)",
    labelpad=8
)

ax_signal_time.spines["top"].set_position(
    ("outward", 0)
)

ax_signal_time.tick_params(
    axis="x",
    pad=4
)


# ------------------------------------------------------------
# Top axis #2 = Distance
# ------------------------------------------------------------

ax_signal_dist = ax_signal.twiny()

ax_signal_dist.set_xlim(ax_signal.get_xlim())

ax_signal_dist.set_xticks(ax_signal.get_xticks())

ax_signal_dist.set_xticklabels(
    [
        f"{SPEED_OF_SOUND * (x / FS) / 2:.2f}"
        for x in ax_signal.get_xticks()
    ]
)

ax_signal_dist.set_xlabel(
    f"Range / Distance (m) — round-trip, "
    f"c = {SPEED_OF_SOUND:g} m/s",
    labelpad=8
)

ax_signal_dist.spines["top"].set_position(
    ("outward", 55)
)

ax_signal_dist.tick_params(
    axis="x",
    pad=4
)

plt.tight_layout()

signal_plot_path = os.path.join(plot_dir, f"{plot_base}_raw_signal.png")
fig_signal.savefig(signal_plot_path, dpi=150, bbox_inches="tight")
print(f"Saved raw signal plot to: {signal_plot_path}")


# ============================================================
# 10. PLOT MATCHED FILTER
# ============================================================

# Sample numbers corresponding to matched-filter output
mf_samples = np.arange(len(matched_filter))

fig, ax = plt.subplots(figsize=(12, 6))

# ------------------------------------------------------------
# Main plot
# ------------------------------------------------------------

ax.plot(
    mf_samples,
    np.abs(matched_filter),
    label="|Matched filter|"
)

ax.plot(
    mf_samples,
    envelope,
    label="Hilbert envelope"
)

# Detected peak
ax.plot(
    peak_index,
    peak_magnitude,
    "ro",
    label=f"Peak = sample {peak_index}"
)


# ------------------------------------------------------------
# Bottom axis = Sample Number
# ------------------------------------------------------------

ax.set_xlabel("Sample Number")
ax.set_ylabel("|Correlation|")
ax.set_title("Matched Filter Output")

# Force x-axis to start at 0
ax.set_xlim(0, len(matched_filter) - 1)

ax.grid(True)


# ============================================================
# TOP AXIS #1 = TIME
# ============================================================

ax_time = ax.twiny()

ax_time.set_xlim(ax.get_xlim())

ax_time.set_xticks(ax.get_xticks())

ax_time.set_xticklabels(
    [
        f"{(x / FS) * 1000:.1f}"
        for x in ax.get_xticks()
    ]
)

ax_time.set_xlabel(
    "Time (ms)",
    labelpad=8
)

ax_time.spines["top"].set_position(
    ("outward", 0)
)

ax_time.tick_params(
    axis="x",
    pad=4
)


# ============================================================
# TOP AXIS #2 = DISTANCE
# ============================================================

ax_dist = ax.twiny()

ax_dist.set_xlim(ax.get_xlim())

ax_dist.set_xticks(ax.get_xticks())

ax_dist.set_xticklabels(
    [
        f"{SPEED_OF_SOUND * (x / FS) / 2:.2f}"
        for x in ax.get_xticks()
    ]
)

ax_dist.set_xlabel(
    f"Range / Distance (m) — round-trip, "
    f"c = {SPEED_OF_SOUND:g} m/s",
    labelpad=8
)

ax_dist.spines["top"].set_position(
    ("outward", 55)
)

ax_dist.tick_params(
    axis="x",
    pad=4
)


# ============================================================
# DISPLAY
# ============================================================

ax.legend()

plt.tight_layout()

mf_plot_path = os.path.join(plot_dir, f"{plot_base}_matched_filter.png")
fig.savefig(mf_plot_path, dpi=150, bbox_inches="tight")
print(f"Saved matched filter plot to: {mf_plot_path}")

plt.show()
