import ctypes
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import adc2mV, assert_pico_ok


# =====================================================
# INITIALIZATION
# =====================================================

def initialize_scope():

    chandle = ctypes.c_int16()
    status = {}

    status["openunit"] = ps.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])

    return chandle, status


# =====================================================
# CHANNEL SETUP
# =====================================================


def configure_channels(chA_range=7): # 2V range 
# Set up channel A
# Range: PS4000_200MV = 5 (200mV range to see 100mV trigger)
# Coupling: PS4000_DC = 1
    status["setChA"] = ps.ps4000SetChannel(
        chandle,
        0,          # Channel A
        1,          # Enable
        1,          # DC coupling
        chA_range
    )
    assert_pico_ok(status["setChA"])
# Disable Channel B (not needed - enables full sampling rate on Channel A)
    status["setChB"] = ps.ps4000SetChannel(
        chandle,
        1,          # Channel B
        0,          # Disable
        1,
        0
    )
    assert_pico_ok(status["setChB"])

    status["setBW"] = ps.ps4000SetBwFilter(chandle, 0, 0)
    assert_pico_ok(status["setBW"])


# =====================================================
# DATA ACQUISITION
# =====================================================

def acquire_data(chandle,
                 num_samples=50000,
                 timebase=8):

    status = {}

    time_interval = ctypes.c_int32()
    returned_samples = ctypes.c_int32()

    status["GetTimebase"] = ps.ps4000GetTimebase(
        chandle,
        timebase,
        num_samples,
        ctypes.byref(time_interval),
        ctypes.byref(returned_samples),
        0
    )

    assert_pico_ok(status["GetTimebase"])

    status["RunBlock"] = ps.ps4000RunBlock(
        chandle,
        0,
        num_samples,
        timebase,
        None,
        0,
        None,
        None
    )

    assert_pico_ok(status["RunBlock"])

    ready = ctypes.c_int16(0)

    while ready.value == 0:
        ps.ps4000IsReady(chandle, ctypes.byref(ready))

    buffer = (ctypes.c_int16 * num_samples)()

    status["SetDataBuffer"] = ps.ps4000SetDataBuffer(
        chandle,
        ps.PS4000_CHANNEL["PS4000_CHANNEL_A"],
        buffer,
        num_samples
    )

    assert_pico_ok(status["SetDataBuffer"])

    overflow = ctypes.c_int16()

    status["GetValues"] = ps.ps4000GetValues(
        chandle,
        0,
        ctypes.byref(returned_samples),
        1,
        0,
        0,
        ctypes.byref(overflow)
    )

    assert_pico_ok(status["GetValues"])

    maxADC = ctypes.c_int16(32767)

    voltage = adc2mV(buffer,
                     ps.PS4000_RANGE["PS4000_5V"],
                     maxADC)

    voltage = np.array(voltage)

    fs = 1 / (time_interval.value * 1e-9)

    t = np.arange(len(voltage)) / fs

    return t, voltage, fs


# =====================================================
# SAVE CSV
# =====================================================

def save_csv(filename, t, voltage):

    data = np.column_stack((t, voltage))

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="Time(s),Voltage(mV)",
        comments=""
    )

    print(f"Saved to {filename}")


# =====================================================
# FFT
# =====================================================

def compute_fft(signal, fs):

    N = len(signal)

    window = np.hanning(N)

    signal = signal * window

    fft = np.fft.rfft(signal)

    freq = np.fft.rfftfreq(N, 1/fs)

    magnitude = np.abs(fft) / N

    return freq, magnitude


# =====================================================
# PSD
# =====================================================

def compute_psd(signal, fs):

    freq, psd = welch(
        signal,
        fs=fs,
        window='hann',
        nperseg=4096,
        scaling='density'
    )

    return freq, psd


# =====================================================
# PLOTTING
# =====================================================

def plot_results(t, signal,
                 fft_freq, fft_mag,
                 psd_freq, psd):

    plt.figure(figsize=(12,8))

    plt.subplot(311)
    plt.plot(t, signal)
    plt.title("Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.grid()

    plt.subplot(312)
    plt.plot(fft_freq, fft_mag)
    plt.title("FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.subplot(313)
    plt.semilogy(psd_freq, psd)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("V²/Hz")
    plt.grid()

    plt.tight_layout()
    plt.show()


# =====================================================
# CLEANUP
# =====================================================

def close_scope(chandle):

    ps.ps4000Stop(chandle)
    ps.ps4000CloseUnit(chandle)


# =====================================================
# MAIN
# =====================================================

def main():

    chandle, status = initialize_scope()

    configure_channels(chandle)

    t, signal, fs = acquire_data(
        chandle,
        num_samples=50000,
        timebase=8
    )

    save_csv("capture.csv", t, signal)

    fft_freq, fft_mag = compute_fft(signal, fs)

    psd_freq, psd = compute_psd(signal, fs)

    plot_results(
        t,
        signal,
        fft_freq,
        fft_mag,
        psd_freq,
        psd
    )

    close_scope(chandle)


if __name__ == "__main__":
    main()