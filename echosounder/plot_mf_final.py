'''this code plots both rx buffer and mf similar to the echosounder console'''

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


# ============================================================
# 1. READ RAW ADC DATA
# ============================================================

filename = "1xGain_23m_50kSamples.txt"

adc_data = np.loadtxt(filename)

print(f"Number of ADC samples: {len(adc_data)}")


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

time = np.arange(len(signal)) / FS

plt.figure(figsize=(12, 5))

plt.plot(time * 1e6, signal)

plt.xlabel("Time (µs)")
plt.ylabel("Voltage (V)")
plt.title("Received ADC Signal")

plt.grid(True)
plt.tight_layout()


# ============================================================
# 10. PLOT TEMPLATE
# ============================================================

'''plt.figure(figsize=(12, 5))

plt.plot(
    t_template * 1e6,
    template
)

plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("Normalized 110 kHz Template")

plt.grid(True)
plt.tight_layout()'''


## ============================================================
# 11. PLOT MATCHED FILTER
# ============================================================

SPEED_OF_SOUND = 1500.0  # m/s

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
plt.show()
