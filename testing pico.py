import ctypes
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import assert_pico_ok

chandle = ctypes.c_int16()

status = ps.ps4000OpenUnit(ctypes.byref(chandle))
assert_pico_ok(status)

print("Connected!")

ps.ps4000CloseUnit(chandle)