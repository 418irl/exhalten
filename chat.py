import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 2000          # Sampling frequency (Hz)
f = 50             # Signal frequency (Hz)
duration = 0.5     # Duration (s)
snr_db = 15        # Desired SNR (dB)

# Generate clean sine wave
t = np.arange(0, duration, 1/fs)
clean_signal = np.sin(2 * np.pi * f * t)

# Calculate signal power
signal_power = np.mean(clean_signal**2)

# Compute required noise power for desired SNR
noise_power = signal_power / (10**(snr_db / 10))

# Generate Gaussian noise
noise = np.random.normal(0, np.sqrt(noise_power), len(t))

# Create noisy signal
noisy_signal = clean_signal + noise

# Compute RMS of noisy signal
rms_noisy = np.sqrt(np.mean(noisy_signal**2))

# Print RMS value
print(f"Measured RMS of noisy signal: {rms_noisy:.4f}")

# Plot signals
plt.figure(figsize=(10, 5))
plt.plot(t, clean_signal, label='Clean Signal')
plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.7)

plt.title('50 Hz Sine Wave with Gaussian Noise (SNR ≈ 15 dB)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()