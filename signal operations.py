import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 1000  # Sampling frequency (Hz)
f = 5  # Signal frequency (Hz)
t = np.linspace(0, 1, fs)  # Time vector for 1 second

# Generating a sine wave
signal = np.sin(2 * np.pi * f * t)

# Plotting the signal
plt.plot(t, signal)
plt.title(f'Sine Wave with Frequency {f} Hz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Signal scaling
scaled_signal = 2 * signal

# Signal addition (with another sine wave of different frequency)
f2 = 10  # Frequency of second sine wave
signal2 = np.sin(2 * np.pi * f2 * t)
added_signal = signal + signal2

# Signal time-shifting
shifted_signal = np.sin(2 * np.pi * f * (t - 0.1))  # Shift by 0.1 seconds

# Plot original and modified signals
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, scaled_signal)
plt.title('Scaled Signal (Amplitude x2)')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, added_signal)
plt.title('Added Signal (5 Hz + 10 Hz)')
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, shifted_signal)
plt.title('Time-Shifted Signal (Shift by 0.1 s)')
plt.grid()

plt.tight_layout()
plt.show()