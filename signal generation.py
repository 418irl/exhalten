import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt

freq = 2 # signal frequency / 2 cycles per 1 second 
# freq = 1 # 1 cycle per 1 second
fs= 1000 #sampling frequency
amp = 2 #amplitude
#x axis scale of graph: time for 2 s
time = np.linspace(0, 2, fs) 

#sine wave
sin0 = amp*np.sin(2*np.pi*freq*time)

#square wave with duty cycle 30% and 50%
sq0 = amp * np.sign(sin0)
#sq0 = amp*sg.square(2*np.pi*freq*time, duty=0.3)
sq1 = amp*sg.square(2*np.pi*freq*time, duty=0.5)

fig, a = plt.subplots(3, 1, figsize=(10, 6))
a[0].plot(time, sin0)
a[1].set_title('sin wave gen')
a[0].set_xlabel('Time (s)')
a[0].set_ylabel('Amplitude')
a[0].grid()

a[1].plot(time, sq0)
a[1].set_title('square wave gen')
a[1].set_xlabel('Time (s)')
a[1].set_ylabel('Amplitude')
a[1].grid()

a[2].plot(time, sq1)
a[2].set_title('square wave gen')
a[2].set_xlabel('Time (s)')
a[2].set_ylabel('Amplitude')
a[2].grid()

plt.tight_layout()
plt.show()
