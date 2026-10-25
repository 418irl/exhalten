import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogSingleChannelReader


device = "Dev1"             # device name in NI MAX (USB-6366)
ai_channel = "ai0"
ai_terminal = TerminalConfiguration.DIFF
ai_range = 1.0               # +/- V; lower it (e.g. 1.0 or 0.2) for small noise signals

sample_rate = 2_000_000       # 2 MS/s (max per channel on USB-6366)
duration = 1               # record length (s)
num_samples = int(sample_rate * duration)

name="resistor_2M_100x_chargeoff"

file_name = f"23102026_{name}.csv"

with nidaqmx.Task() as ai_task:

    ai_task.ai_channels.add_ai_voltage_chan(
        f"{device}/{ai_channel}",
        terminal_config=ai_terminal,
        min_val=-ai_range,
        max_val=ai_range,
    )
    ai_task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=num_samples,
    )

    reader = AnalogSingleChannelReader(ai_task.in_stream)
    ai_data = np.zeros(num_samples, dtype=np.float64)

    print("Acquiring...")
    ai_task.start()
    reader.read_many_sample(
        ai_data,
        number_of_samples_per_channel=num_samples,
        timeout=duration + 5,
    )
    ai_task.wait_until_done()

time = np.arange(num_samples) / sample_rate

print("Acquisition complete.")


np.savetxt(file_name, ai_data, delimiter=",")
print("Saving complete")

