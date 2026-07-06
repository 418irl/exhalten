import ctypes
import numpy as np
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


def configure_channels(chA_range=7): # 200mV range (can see 100mV trigger level)
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

def configure_trigger(trigger_adc=30000):
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

    return voltage, cmaxSamples.value

# Create time array in milliseconds for better readability
def create_time_axis(sample_count, interval_ns):

    time_ns = np.arange(sample_count) * interval_ns

    return time_ns / 1e6

def plot_waveform(time_ms, voltage):

    plt.figure(figsize=(10,5))
    plt.plot(time_ms, voltage)

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f'PicoScope 4262 - Channel A @ {actual_fs/1e6:.2f} MHz, Duration: {duration*1000} ms')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def close_scope():

    status["stop"] = ps.ps4000Stop(chandle)
    assert_pico_ok(status["stop"])

    status["close"] = ps.ps4000CloseUnit(chandle)
    assert_pico_ok(status["close"])

    print("Scope closed")

def main():

    open_scope()

    configure_channels()

    configure_trigger()

    pre, post, samples = configure_acquisition()

    timebase = 0

    interval_ns, actual_fs, oversample = get_timebase(timebase, samples)

    capture_block(pre, post, timebase, oversample)

    voltage, sample_count = read_data(samples, 7)

    time_ms = create_time_axis(sample_count, interval_ns)

    plot_waveform(time_ms, voltage)

    close_scope()

    print(f"Actual Sampling Rate = {actual_fs/1e6:.2f} MHz")


if __name__ == "__main__":
    main()  
#*******************
