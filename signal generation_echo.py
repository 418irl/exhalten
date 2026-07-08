import numpy as np
import matplotlib.pyplot as plt

# Barker codes
BARKER_CODES = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
}

fc = 110e3      # Carrier frequency (Hz) 110kHz
fs = 1e6        # Sampling frequency (Hz) 1MHz


def barker_pulse(code_length, n_cycles_per_chip=1):

    code = BARKER_CODES[code_length]
    #chip_duration = 1.0 / len(code)
    chip_duration = n_cycles_per_chip / fc   #T=no of cycles/carrier frequency = 1/110k=9.09us

    n_chip = int(round(chip_duration * fs)) # no of samples per chip N=T*fs 
    t_chip = np.arange(n_chip) / fs # time for 1 chip 
    chips = [bit * np.sin(2*np.pi*fc*t_chip) for bit in code] #bit*sin=>+_sin ie phase shift
    sig = np.concatenate(chips) #to get one continuous waveform

    t = np.arange(len(sig)) / fs #corresponds to entire barker signal

    return t, sig


#lengths = [2,3,4,5,7,11,13]

#to plot overlapping
'''
plt.figure(figsize=(10,6))
for L in lengths:
    t, sig = barker_pulse(L)
    # FFT
    N = len(sig)

    fft = np.fft.rfft(sig)

    freq = np.fft.rfftfreq(N, d=1/fs)

    # Magnitude
    fft_mag = np.abs(fft)

    # Normalize (optional)
    fft_mag /= np.max(fft_mag)
    plt.plot(freq/1e3, fft_mag, label=f"Barker-{L}")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized Magnitude")
    plt.title("FFT of Barker Pulse")
    plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''


#plot everything as subplots
'''
fig, a = plt.subplots(len(lengths), 1, figsize=(10,12), sharex=True)

for a,L in zip(a, lengths):
    t, sig = barker_pulse(L)
    
    # FFT
    N = len(sig)
    fft = np.fft.rfft(sig)

    freq = np.fft.rfftfreq(N, d=1/fs)

    # Magnitude
    fft_mag = np.abs(fft)

    # Normalize (optional)
    fft_mag /= np.max(fft_mag)

    a.plot(freq/1e3, fft_mag)
    a.set_title(f"Barker-{L}")
    #a.set_xlabel("Frequency (kHz)")
    #a.set_ylabel("Normalized Magnitude")
    #a.set_title("FFT of Barker Pulse")
a.set_xlabel("Frequency (kHz)")
fig.supylabel("Normalized Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()'''


t, sig = barker_pulse(13)
# FFT
N = len(sig)
fft = np.fft.rfft(sig)
freq = np.fft.rfftfreq(N, d=1/fs)

# Magnitude
fft_mag = np.abs(fft)

# Normalize (optional)
fft_mag /= np.max(fft_mag)

fig,a=plt.subplots(2,1)
a[0].plot(freq/1e3, fft_mag)
a[1].plot(t/1e4,sig)

'''plt.plot(freq/1e3, fft_mag)
plt.xlabel("Frequency (kHz)")
plt.ylabel("Normalized Magnitude")
plt.title("FFT of Barker Pulse")
plt.legend()'''
plt.grid(True)
plt.tight_layout()
plt.show()