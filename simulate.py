import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import librosa
from scipy.io import wavfile as wav

# The desired reverberation time and dimensions of the room
rt60_tgt = 0.3  # seconds
room_dim = [10, 7.5, 3.5]  # meters

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
fs, audio1 = wavfile.read("inputs/sitar.wav")
fs, audio2 = wavfile.read("inputs/piano.wav")

audio1 = audio1[1000000:2000000, :1]
audio1 = audio1.reshape(audio1.shape[0])
audio1 = audio1/5

audio2 = audio2[1000000:2000000, :1]
audio2 = audio2.reshape(audio1.shape[0])

print(audio1.shape)
print(audio2.shape)

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
)

# place the source in the room
room.add_source([7, 5, 1], signal=audio1, delay=0.)
room.add_source([2, 3, 1], signal=audio2, delay=0.)

# define the locations of the microphones
mic_locs = np.c_[
    # [4.9, 4, 1], 
    # [5.1, 4.1, 1], 
    # [5.1, 3.9, 1], 
    # [5, 4, 1.2], 
    # [5, 4, 0.8],
    # [4.9, 3.9, 1.2],
    [5.1, 4.1, 0.8]
]

# finally place the array in the room
room.add_microphone_array(mic_locs)

# Run the simulation (this will also build the RIR automatically)
room.simulate()

room.mic_array.to_wav(
    f"outputs/mic7_music.wav",
    norm=True,
    bitdepth=np.int16,
)


# print(room.mic_array.signals)
# fs, audio = wavfile.read("outputs/music.wav")
# print(audio.shape)


# # # measure the reverberation time
# # rt60 = room.measure_rt60()
# # print("The desired RT60 was {}".format(rt60_tgt))
# # print("The measured RT60 is {}".format(rt60[1, 0]))

# # # Create a plot
# # plt.figure()

# # # plot one of the RIR. both can also be plotted using room.plot_rir()
# # rir_1_0 = room.rir[1][0]
# # plt.subplot(2, 1, 1)
# # plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
# # plt.title("The RIR from source 0 to mic 1")
# # plt.xlabel("Time [s]")

# # plot signal at microphone 1
# plt.subplot(2, 1, 2)
# print(room.mic_array.signals[1, :])
# # for i, signal in enumerate(room.mic_array.signals):
# #     print(i, signal)
# #     print(np.int16(signal*32768.0))
# #     wav.write("outputs/" + str(i) + ".wav", 44100, np.int16(signal*32768.0))
# plt.plot(room.mic_array.signals[1, :])
# plt.title("Microphone 1 signal")
# plt.xlabel("Time [s]")

# plt.tight_layout()
# plt.show()