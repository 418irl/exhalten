import ctypes
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import adc2mV, assert_pico_ok

# Create chandle and status ready for use
status = {}
chandle = ctypes.c_int16()

# Open PicoScope 4000 series
def open_scope():
    status["openunit"] = ps.ps4000OpenUnit(ctypes.byref(chandle))
    assert_pico_ok(status["openunit"])
    print("PicoScope opened successfully")

def configure_channels(chA_range=3): # 2V range 
    '''Enum	Voltage Range
    0	10 mV
    1	20 mV
    2	50 mV
    3	100 mV
    4	200 mV
    5	500 mV
    6	1 V
    7	2 V
    8	5 V
    9	10 V
    10	20 V'''
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

def configure_trigger(trigger_adc=1638):
    # Calculate trigger threshold for 100mV
    # For 20mV range, max ADC = 32767 corresponds to 20mV
    # 100mV trigger = (100mV / 20mV) * 32767 ADC counts
    # But 100mV > 20mV range, so we need a larger range for trigger
    # Using PS4000_200MV = 5 range: (100mV / 200mV) * 32767 = 16384 ADC counts

    # Set up single trigger
    # Trigger at 100mV on Channel A, rising edge
    # auto Trigger = 0 (disabled - wait indefinitely for trigger)
    status["trigger"] = ps.ps4000SetSimpleTrigger(
        chandle,
        1,              # Enable trigger
        0,              # Channel A
        trigger_adc,
        2,              # Rising edge
        0,
        0
    )
    assert_pico_ok(status["trigger"])

def configure_acquisition(fs=10e6, duration=0.001):
    # Sampling parameters
    # fs = 10 MHz = 10e6 samples/second
    # Duration = 1 ms = 0.001 seconds
    # Total samples = fs * duration = 10e6 * 0.001 = 10,000 samples

    total_samples = int(fs * duration)

    pre_trigger = 0
    post_trigger = total_samples

    return pre_trigger, post_trigger, total_samples
#def calculate_timebase(fs):
    """
    Calculate an approximate PS4000 timebase for a desired
    sampling frequency.

    Parameters
    ----------
    fs : float
        Desired sampling frequency (Hz)

    Returns
    -------
    int
        Approximate timebase
    """
    interval_ns = 1e9 / fs
    timebase = int(round(interval_ns / 8 + 2))
    return max(timebase, 0)

def find_timebase(target_fs, max_samples):

    for tb in range(0, 500):

        interval = ctypes.c_float()
        returned = ctypes.c_int32()

        status = ps.ps4000GetTimebase2(
            chandle,
            tb,
            max_samples,
            ctypes.byref(interval),
            ctypes.c_int16(1),
            ctypes.byref(returned),
            0
        )

        if status == 0:

            fs = 1e9 / interval.value

            if fs <= target_fs:
                print(f"Timebase = {tb}")
                print(f"Actual fs = {fs}")
                return tb

    raise RuntimeError("Couldn't find suitable timebase")
def get_timebase(timebase, max_samples):
    # Calculate timebase for 10 MHz sampling (100ns interval)
    # For PS4000 with 2 channels enabled:
    # - Timebase 0 = 500 MHz (2ns)
    # - Timebase 1 = 250 MHz (4ns)  
    # - Timebase 2 = 125 MHz (8ns)
    # - Timebase 3 and above: interval = (timebase - 2) * 8ns
    # For 100ns interval (10 MHz): (timebase - 2) * 8ns = 100ns
    # Therefore: timebase = 100/8 + 2 = 14.5, round to 14
    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    oversample = ctypes.c_int16(1)

    status["getTimebase2"] = ps.ps4000GetTimebase2(
        chandle,
        timebase,
        max_samples,
        ctypes.byref(timeIntervalns),
        oversample,
        ctypes.byref(returnedMaxSamples),
        0
    )

    assert_pico_ok(status["getTimebase2"])

    actual_fs=1e9 / timeIntervalns.value
    print("Returned interval:", timeIntervalns.value, "ns")
    print("Returned fs:", actual_fs)

    return timeIntervalns.value, actual_fs, oversample

# Run block capture
def capture_block(pre_trigger, post_trigger, timebase, oversample):

    status["runBlock"] = ps.ps4000RunBlock(
        chandle,
        pre_trigger,
        post_trigger,
        timebase,
        oversample,
        None,
        0,
        None,
        None
    )

    assert_pico_ok(status["runBlock"])
    print("Waiting for trigger...")

    # Wait for data collection to finish
    ready = ctypes.c_int16(0)

    while ready.value == 0:
        ps.ps4000IsReady(chandle, ctypes.byref(ready))

    print("Capture complete")

def read_data(max_samples, chA_range):
# Create buffers for data collection (Channel A only)
    bufferAMax = (ctypes.c_int16 * max_samples)()
    bufferAMin = (ctypes.c_int16 * max_samples)()

# Set data buffer location for Channel A only
    status["setDataBuffers"] = ps.ps4000SetDataBuffers(
        chandle,
        0,
        ctypes.byref(bufferAMax),
        ctypes.byref(bufferAMin),
        max_samples
    )

    assert_pico_ok(status["setDataBuffers"])

# Create overflow location
    overflow = ctypes.c_int16()
    cmaxSamples = ctypes.c_int32(max_samples)

# Retrieve data from scope
    status["getValues"] = ps.ps4000GetValues(
        chandle,
        0,
        ctypes.byref(cmaxSamples),
        0,
        0,
        0,
        ctypes.byref(overflow)
    )

    assert_pico_ok(status["getValues"])

# Convert ADC counts to mV (Channel A only)
    maxADC = ctypes.c_int16(32767)

#adc2mVChAMax Voltage
    voltage = adc2mV(bufferAMax, chA_range, maxADC)
    voltage = np.asarray(voltage, dtype=float)
    return voltage, cmaxSamples.value

# Create time array in milliseconds for better readability
def create_time_axis(sample_count, interval_ns):

    time_ns = np.arange(sample_count) * interval_ns

    return time_ns / 1e6

def save_csv(filename, time_ms, voltage):

    data = np.column_stack((time_ms, voltage))

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header="Time(s),Voltage(mV)",
        comments="")

    print(f"Saved to {filename}")

# FFT =====================================================

def compute_fft(voltage, fs):

    N = len(voltage)

    window = np.hanning(N)

    coherent_gain = np.mean(window)

    voltage = voltage * window

    fft = np.fft.rfft(voltage)

    freq = np.fft.rfftfreq(N, 1/fs)

    #magnitude = np.abs(fft) / N
    magnitude = (2 * np.abs(fft)) / (N * coherent_gain)

    magnitude[0] /= 2
    magnitude_db = 20*np.log10(np.maximum(magnitude,1e-20))

    return freq, magnitude_db

# PSD =====================================================

def compute_psd(voltage, fs):
    nperseg = min(4096, len(voltage))
    print(np.min(voltage))
    print(np.max(voltage))
    freq, psd = sg.welch(
        voltage,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        scaling='density'
    )
    psd_db = 10*np.log10(np.maximum(psd,1e-30))

    print("***********************************************")
    df = freq[1] - freq[0]
    peak = np.argmax(psd)
    
    noise_floor_db = np.median(psd_db)
    '''
    signal_bins = slice(peak-2, peak+3)
    signal_power = np.sum(psd[signal_bins]) * df
    mask = np.ones(len(psd), dtype=bool)
    mask[peak-2:peak+3] = False

    noise_power = np.sum(psd[mask]) * df'''

    band = freq <= 100000      # 100 kHz

    noise_power = np.sum(psd[band]) * df

    Vrms = np.sqrt(noise_power)
    
    #signal_power = np.sum(psd[peak-2:peak+3]) * df
    #noise_power = np.sum(psd)*df

    #SNR = 10*np.log10(signal_power/noise_power)
    #Vrms = np.sqrt(np.sum(psd) * df)


    print("noise floor:", noise_floor_db)
    print("noise power:", noise_power)
    #print("signal power:", signal_power)
    #print("SNR:", SNR)
    #print("PSD:", psd)
    print("Vrms:", Vrms)
    print("df =", df)
    print("fs =", fs)
    print("len(psd) =", len(psd))

    print("***********************************************")
    return freq, psd_db

# PLOTTING =====================================================

def plot_results(time_ms, voltage,
                 fft_freq, fft_mag,
                 psd_freq, psd):

    plt.figure(figsize=(12,8))

    plt.subplot(311)
    plt.plot(time_ms, voltage)
    plt.title("Time Domain")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.grid()

    plt.subplot(312)
    plt.plot(fft_freq, fft_mag)
    plt.title("FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()

    plt.subplot(313)
    plt.plot(psd_freq, psd)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("dBmV²/Hz")
    plt.grid()

    plt.tight_layout()
    plt.show()


#def plot_waveform(time_ms, voltage, actual_fs, duration):
    '''
    plt.figure(figsize=(10,5))
    plt.plot(time_ms, voltage)

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f'PicoScope 4262 - Channel A @ {actual_fs/1e6:.2f} MHz, Duration: {duration*1000} ms')
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''

def close_scope():

    status["stop"] = ps.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    status["close"] = ps.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    print("Scope closed")

def main():

    open_scope()

    configure_channels()

    #configure_trigger()

    duration=0.001
    fs = 1e6          # 1 MHz
    pre, post, samples = configure_acquisition(fs,duration)

    print(fs)
    print(duration)
    print(samples)


    
    timebase = find_timebase(1e6, samples)
    #timebase = calculate_timebase(fs)
    print(timebase)

    interval_ns, actual_fs, oversample = get_timebase(timebase, samples)

    capture_block(pre, post, timebase, oversample)

    voltage, sample_count = read_data(samples, 7)

    time_ms = create_time_axis(sample_count, interval_ns)

    print("Requested fs :", fs)
    print("Actual fs    :", actual_fs)
    print("Interval(ns) :", interval_ns)
    print("Samples      :", sample_count)

    print("First five time values:")
    print(time_ms[:5])

    print("Last time value:")
    print(time_ms[-1])

    filename=r"C:\Users\fores\OneDrive\Desktop\xalten\xalten\picoscope_csv\test_noise.csv"
    save_csv(filename, time_ms, voltage)

    fft_freq, fft_mag = compute_fft(voltage, fs)

    psd_freq, psd = compute_psd(voltage, fs)
    
    plot_results(
        time_ms,
        voltage,
        fft_freq,
        fft_mag,
        psd_freq,
        psd
    )


    #plot_waveform(time_ms, voltage, actual_fs, duration)

    close_scope()

    print(f"Actual Sampling Rate = {actual_fs/1e6:.2f} MHz")


if __name__ == "__main__":
    main()  
#*******************