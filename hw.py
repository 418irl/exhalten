import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg

'''
f=50
fs1=1000
fs2=200
fs3=80

t1=np.linspace(0,2,fs1)
t2=np.linspace(0,2,fs2)
t3=np.linspace(0,2,fs3)

y1=np.sin(2*np.pi*f*t1)
y2=np.sin(2*np.pi*f*t2)
y3=np.sin(2*np.pi*f*t3)

fig, a=plt.subplots(3,1)

a[0].plot(t1,y1)
a[0].set_title("1000")
a[1].plot(t2,y2)
a[1].set_title("200")
a[2].plot(t3,y3)
a[2].set_title("80")

plt.tight_layout()
plt.show()
'''


#convolution
'''
x = np.array([1, 1])
h = np.array([1, 2, 1])

n_x = np.arange(0, len(x))  # [0, 1]
n_h = np.arange(0, len(h))  # [0, 1, 2]

n_y = np.arange(n_x[0] + n_h[0], n_x[-1] + n_h[-1] + 1)

y=np.convolve(x,h)
print("conv=",y)

# 4. Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.stem(n_x,x)
plt.title('Original Signal 1')

plt.subplot(3, 1, 2)
plt.stem(n_h,h)
plt.title('Original Signal 2')

plt.subplot(3, 1, 3)
plt.stem(n_y, y, 'r')
plt.title('Convolved Signal')
plt.xlabel('Sample Index')

plt.tight_layout()
plt.show()
'''

# 1. Generate a clean signal + high-frequency noise
t = np.linspace(0, 1, 200)  # 200 time steps over 1 second

# Clean underlying signal (Low Frequency: 3 Hz sine wave)
clean_signal = np.sin(2 * np.pi * 3 * t)

# High-frequency noise
noise = 0.4 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.random.randn(len(t))

# The messy composite signal that enters our filters
noisy_signal = clean_signal + noise

# 2. Design the Filter Shapes (Impulse Responses)
# LPF: A smoothing window (moving average)
# This shape blurs out rapid wiggles
lpf_kernel = np.ones(11) / 11  

# HPF: A differentiating window
# This shape cancels out smooth trends and keeps sharp changes
hpf_kernel = np.array([-1, 2, -1]) 

# 3. Apply Filtering using Convolution
# np.convolve performs the exact "flip, slide, multiply, sum" machinery
# mode='same' ensures the output array is the same length as the input
lpf_output = np.convolve(noisy_signal, lpf_kernel, mode='same')
hpf_output = np.convolve(noisy_signal, hpf_kernel, mode='same')

# 4. Plot the Results
plt.figure(figsize=(12, 8))

# Top Plot: Low-Pass Filter (Smoothing)
plt.subplot(2, 1, 1)
plt.plot(t, noisy_signal, color='lightgray', label='Noisy Input Signal')
plt.plot(t, clean_signal, 'g--', label='Original Clean Signal (3 Hz)')
plt.plot(t, lpf_output, color='blue', linewidth=2, label='LPF Output (Smooth)')
plt.title('Low-Pass Filter (Blocks High-Frequency Noise)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Bottom Plot: High-Pass Filter (Edge/Noise Detection)
plt.subplot(2, 1, 2)
plt.plot(t, noisy_signal, color='lightgray', label='Noisy Input Signal')
plt.plot(t, hpf_output, color='red', linewidth=1.5, label='HPF Output (Sharp Changes Only)')
plt.title('High-Pass Filter (Blocks Low-Frequency Base, Passes Noise/Edges)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()