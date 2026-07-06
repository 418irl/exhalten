import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg

f=2
fs = 1000
ts=2
snr=15
t = np.arange(0, ts, 1/fs)

#y=np.zeros(len(t))
#y[t>=0]=1
'''
#impulse function
y[np.isclose(t,0,atol=0.000001)]=1

#triangle function
t=np.arange(-5,5,0.1) 
y=(1-abs(t))*(abs(t)<1)


# Set up t over a larger range
t = np.arange(-5, 5, 0.1)

# Define the width w of each half-cycle
w = 2  # Example width for each cycle

# Use modulo to create multiple cycles
y = np.zeros(len(t))
y[((t % w) > (-w/2)) & ((t % w) <= 0)] = 1  # Positive part of each cycle
y[((t % w) < (w/2)) & ((t % w) > 0)] = -1   # Negative part of each cycle
'''

#y=np.sin(2*np.pi*f*t)
y = 2*sg.square(2*np.pi*f*t, duty=0.5)

#to shift function
k=int(input("enter shifting factor:"))
t1 = [x - k for x in t]

clean = np.sin(2*np.pi*f*t)
psignal=np.mean(clean**2)

noise = 0.18*np.random.randn(len(t))
pnoise=(10**(snr/10))/psignal

noisy = clean + noise
rms=np.sqrt(np.mean(noisy**2))


'''
print("rms value=",rms)
plt.plot(t, clean,label='Clean Signal')
plt.plot(t,noisy, label='Noisy Signal', alpha=0.7)
plt.title('50 Hz Sine Wave with Gaussian Noise (SNR ≈ 15 dB)')'''
plt.plot(t,y)
#plt.plot(t1,y)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.show()

