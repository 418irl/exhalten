from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
# Define two simple signals
x = np.array([1, 2, 3])
h = np.array([0, 1, 0.5])
fs= 1000
t = np.linspace(0, 1, fs)
# Perform linear convolution
y = np.convolve(x, h)

print("Linear Convolution Result:", y)

#convolution without np.convolve
#x1 = [int(i) for i in input("Enter elements of the input sequence : ").split(',')]
#h1 = [int(i) for i in input("Enter elements of the impulse response : ").split(',')]

# N=P+Q-1
N = len(x) + len(h) - 1

y1 = [0] * N
# Compute the linear convolution using the direct mathematical expression
for n in range(N):
    for k in range(len(h)):
        if n - k >= 0 and n - k < len(x):
            y1[n] += x[n - k] * h[k]

print("Linear Convolution using Mathematical expression : ", y1)

'''
# Generating a signal for DFT
f1 = 50  # Frequency of first sine wave (50 Hz)
f2 = 120  # Frequency of second sine wave (120 Hz)
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# Compute DFT
signal_dft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/fs)

# Plot the DFT magnitude
plt.plot(freqs[:len(freqs)//2], np.abs(signal_dft[:len(signal_dft)//2]))
plt.title('DFT of the Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.show()
'''