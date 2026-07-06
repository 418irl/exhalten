'''import numpy as np
import matplotlib.pyplot as plt

def welch_from_scratch(x, fs, nperseg, noverlap, nfft=None):
    """
    Mathematical implementation of Welch's Method.
    x       : Input signal array
    fs      : Sampling frequency
    nperseg : Length of each segment (M)
    noverlap: Number of points to overlap between segments
    nfft    : Number of FFT points (defaults to nperseg)
    """
    if nfft is None:
        nfft = nperseg
        
    N = len(x)
    M = nperseg
    D = M - noverlap  # Shift distance between segments
    
    # 1. Define the Window (e.g., Hann window) and its scaling factor U
    w = np.hanning(M)
    U = np.sum(w**2) / M
    
    # Calculate total number of segments K
    K = int(np.floor((N - M) / D)) + 1
    
    # Array to accumulate periodograms (only need the positive frequencies)
    num_freqs = nfft // 2 + 1
    psd_accumulator = np.zeros(num_freqs)
    
    # 2 & 3. Loop through segments, window, and compute individual FFTs
    for i in range(K):
        start = i * D
        end = start + M
        segment = x[start:end]
        
        # Apply window
        windowed_segment = segment * w
        
        # Compute FFT and take the magnitude squared
        fft_values = np.fft.rfft(windowed_segment, n=nfft)
        periodogram = (np.abs(fft_values) ** 2) / (fs * M * U)
        
        # Scale for one-sided PSD (multiply by 2 for positive freqs, except DC/Nyquist)
        periodogram[1:-1] *= 2
        
        psd_accumulator += periodogram
        
    # 4. Average the results
    psd_final = psd_accumulator / K
    frequencies = np.fft.rfftfreq(nfft, d=1/fs)
    
    return frequencies, psd_final




# ---------- Generate Test Signal ----------
fs = 1000                  # Sampling frequency (Hz)
duration = 2               # seconds

t = np.arange(0, duration, 1/fs)

# 50 Hz sine wave
signal = np.sin(2*np.pi*50*t)

# Add Gaussian noise
noise = 0.5 * np.random.randn(len(t))

x = signal + noise

# ---------- Compute PSD ----------
freqs, psd = welch_from_scratch(
    x,
    fs=fs,
    nperseg=256,
    noverlap=128,
    nfft=256
)

# ---------- Plot Time Signal ----------
plt.figure(figsize=(10,4))
plt.plot(t, x)
plt.title("Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# ---------- Plot PSD ----------
plt.figure(figsize=(10,4))
plt.semilogy(freqs, psd)
plt.title("Welch PSD (From Scratch)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (Power/Hz)")
plt.grid(True)

plt.show()'''

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Welch PSD (Your implementation)
# -----------------------------------------------------
def welch_from_scratch(x, fs, nperseg, noverlap, nfft=None):

    if nfft is None:
        nfft = nperseg

    N = len(x)
    M = nperseg
    D = M - noverlap

    # Hann window
    w = np.hanning(M)

    # Window power normalization
    U = np.sum(w**2) / M

    # Number of segments
    K = int(np.floor((N - M) / D)) + 1

    # PSD accumulator
    num_freqs = nfft // 2 + 1
    psd_accumulator = np.zeros(num_freqs)

    for i in range(K):

        start = i * D
        end = start + M

        segment = x[start:end]

        # Windowing
        windowed = segment * w

        # FFT
        X = np.fft.rfft(windowed, n=nfft)

        # Periodogram
        Pxx = (np.abs(X)**2) / (fs * M * U)

        # One-sided correction
        Pxx[1:-1] *= 2

        psd_accumulator += Pxx

    psd = psd_accumulator / K

    f = np.fft.rfftfreq(nfft, d=1/fs)

    return f, psd


# -----------------------------------------------------
# Standard Periodogram
# -----------------------------------------------------
def periodogram_from_scratch(x, fs):

    N = len(x)

    X = np.fft.rfft(x)

    Pxx = (np.abs(X)**2) / (fs * N)

    # Convert to one-sided PSD
    Pxx[1:-1] *= 2

    f = np.fft.rfftfreq(N, d=1/fs)

    return f, Pxx


# -----------------------------------------------------
# Generate Test Signal
# -----------------------------------------------------
fs = 1000            # Sampling frequency
duration = 5         # seconds

t = np.arange(0, duration, 1/fs)

# Two sine waves
signal = (
    1.0*np.sin(2*np.pi*50*t) +
    0.5*np.sin(2*np.pi*120*t)
)

# Add Gaussian noise
noise = 0.7*np.random.randn(len(t))

x = signal + noise


# -----------------------------------------------------
# Compute PSDs
# -----------------------------------------------------
f_periodogram, psd_periodogram = periodogram_from_scratch(x, fs)

f_welch, psd_welch = welch_from_scratch(
    x,
    fs,
    nperseg=256,
    noverlap=128,
    nfft=256
)

# -----------------------------------------------------
# Plot Time Signal
# -----------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(t, x)
plt.title("Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# -----------------------------------------------------
# Standard Periodogram
# -----------------------------------------------------
plt.figure(figsize=(10,4))
plt.semilogy(f_periodogram, psd_periodogram)
plt.title("Standard Periodogram")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (Power/Hz)")
plt.xlim(0,250)
plt.grid(True)

# -----------------------------------------------------
# Welch PSD
# -----------------------------------------------------
plt.figure(figsize=(10,4))
plt.semilogy(f_welch, psd_welch)
plt.title("Welch PSD")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (Power/Hz)")
plt.xlim(0,250)
plt.grid(True)

# -----------------------------------------------------
# Comparison
# -----------------------------------------------------
plt.figure(figsize=(10,5))
plt.semilogy(f_periodogram, psd_periodogram,
             label="Standard Periodogram",
             alpha=0.6)

plt.semilogy(f_welch, psd_welch,
             linewidth=2,
             label="Welch PSD")

plt.title("PSD Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (Power/Hz)")
plt.xlim(0,250)
plt.grid(True)
plt.legend()

plt.show()