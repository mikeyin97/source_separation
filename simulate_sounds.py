import time
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra

def test_func(cat1, cat2, cat2fn):
    return cat1, cat2

def func(cat1, cat2, cat2fn):
    rt60_tgt = 0.3  # seconds
    room_dim = [10, 10, 3]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # microphone locations
    mic_locs = np.c_[
        [4.9, 4, 1], 
        [5.1, 4.1, 1], 
        [5.1, 3.9, 1], 
        [5, 4, 1.2], 
        [5, 4, 0.8],
        [4.9, 3.9, 1.2],
        [5.1, 4.1, 0.8]
    ]

    for i in range(100):
        rand1 = np.random.randint(0, len(cat2fn[cat1]))
        rand2 = np.random.randint(0, len(cat2fn[cat2]))
        fn1 = cat2fn[cat1][rand1]
        fn2 = cat2fn[cat2][rand2]

        fs, audio1 = wavfile.read("inputs/ESC-50-master/audio/" + fn1)
        fs, audio2 = wavfile.read("inputs/ESC-50-master/audio/" + fn2)

        min_len = min(audio1.shape[0], audio2.shape[0])

        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        for i in range(5):
            room = pra.ShoeBox(
                room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
            )
            values = np.random.random(6) * 2.0 - 1.0
            loc1 = [values[0] + 5, values[1] + 5, values[2] + 1.5]
            loc2 = [values[3] + 5, values[4] + 5, values[5] + 1.5]
            
            room.add_source(loc1, signal=audio1, delay=0.)
            room.add_source(loc2, signal=audio2, delay=0.)
            room.add_microphone_array(mic_locs)
            room.simulate()
            
            filename = fn1 + " " + fn2 + " " + str(loc1) + str(loc2) + ".wav"
            room.mic_array.to_wav(
                    f"outputs/combined/" + filename,
                    norm=True,
                    bitdepth=np.int16,
            )
    # choice of 30 x 30 = 900 combs
    # x 5 locations each
    