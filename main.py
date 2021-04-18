from scipy.io import wavfile as wav
from helpers.hyperboloid import *
from helpers.lookup_helpers import *
from helpers.separation import *
from helpers.signals import *
from helpers.tdoa import *
from collections import Counter
import librosa
import librosa.display
import warnings


warnings.simplefilter('ignore', wav.WavFileWarning)

mic_locs = {}
# z is height, y is depth
mic_locs["mic1"] = np.array([0,0,0]) 
mic_locs["mic2"] = np.array([0,0.04,0])
mic_locs["mic3"] = np.array([-1*np.sqrt(0.04**2 - 0.02**2),0.02,0])
mic_locs["mic4"] = np.array([-1*np.sqrt(0.04**2 - 0.02**2),-0.02,0])
mic_locs["mic5"] = np.array([0,-0.04,0])
mic_locs["mic6"] = np.array([np.sqrt(0.04**2 - 0.02**2),-0.02,0])
mic_locs["mic7"] = np.array([np.sqrt(0.04**2 - 0.02**2),0.02,0])
mic_locs_list = list(mic_locs.values())

def load_from_path(folder):
    mics = {}
    mics["mic1"], sample_rate = librosa.load(folder + "Audio Track.wav")
    mics["mic2"], sample_rate = librosa.load(folder + "Audio Track-2.wav")
    mics["mic3"], sample_rate = librosa.load(folder + "Audio Track-3.wav")
    mics["mic4"], sample_rate = librosa.load(folder + "Audio Track-4.wav")
    mics["mic5"], sample_rate = librosa.load(folder + "Audio Track-5.wav")
    mics["mic6"], sample_rate = librosa.load(folder + "Audio Track-6.wav")
    mics["mic7"], sample_rate = librosa.load(folder + "Audio Track-7.wav")
    return mics, sample_rate

def vstack_mics(mics, normalize = False):
    data = []
    for i in mics:
        data.append(mics[i])
    data = np.vstack(data)
    data = np.asmatrix(data)
    if normalize:
        data = data/32768.0
    return data

def plot_with_mics(locs, mics_loc_list, count = False, sv = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')   
    if count:
        loc_counter = Counter(locs)
        for i in loc_counter:
            if i is not None:
                ax.scatter(i[0], i[1], i[2], c = "blue", s = 10*np.log(loc_counter[i]))
    else:
        loc_counter = locs
        for i in loc_counter:
            ax.scatter(i[0], i[1], i[2], c = "blue")

    for loc in mic_locs_list:
        ax.scatter(loc[0], loc[1], loc[2], c = "black")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    if sv:
        plt.savefig(sv)

    plt.show()

def mic_to_wavs(source, fname):
    s = np.hstack(source)

    for i in range(s.shape[0]):
        src = s[i]
        src = src.astype(np.float32)
        if i == 0:
            wav.write(fname + '/Audio Track.wav', 44100, src.T)
        else:
            wav.write(fname + '/Audio Track-' + str(i+1) + '.wav', 44100, src.T)
        
def final_res_to_wavs(source, fname):
    y = np.hstack(final_data)
    src1 = y[0]
    src1 = src1
    src1 = src1.astype(np.float32)
    wav.write(fname + 'res1.wav', 44100, src1.T)

    src2 = y[1]
    src2 = src2
    src2 = src2.astype(np.float32)
    wav.write(fname + 'res2.wav', 44100, src2.T)
    
def plt_wavs(wavs, sr, pts = None, sv = None):
    fig, axs = plt.subplots(3,3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    count = 0
    for i in wavs:
        # axs[count//3, count%3].plot()
        plt.sca(axs[count//3, count%3])
        librosa.display.waveplot(wavs[i],sr=sr, max_points=50000.0, x_axis='time', offset=0.0)
        if pts != None:
            for p in pts[i]:
                plt.scatter(p[0]/sr, p[1], color="red")
        count += 1
    if sv:
        plt.savefig(sv)
    plt.show()
    

def find_peaks(mics, height, distance):
    all_peaks = {}
    for m in mics:
        i = 1
        peaks = []
        x = mics[m]
        while i < x.shape[0] - 1:
            if abs(x[i]) >= height:
                if abs(x[i]) >= abs(x[i-1]) and abs(x[i]) >= abs(x[i+1]):
                    peaks.append([i, x[i]])
                    i += distance
                else:
                    i += 1
            else:
                i += 1
        all_peaks[m] = peaks
    return all_peaks

def normalize(mics):
    new_mics = {}
    for m in mics:
        new_mics[m] = mics[m] / np.max(mics[m])
    return new_mics

if __name__ == "__main__":
    # SELECT FOLDER
    folder = "./test/talking/1/"
    # folder = "./res1/src1/"
    # folder = "./res1/src2/"
    
    # LOAD DATA
    frames = 2**12
    mics, sample_freq = load_from_path(folder)
    data = vstack_mics(mics, normalize = True)
    
    # PLOT DATA
    localized_pts = []
    pts = find_peaks(mics, 0.14, 10000)
    plt_wavs(mics, sample_freq, pts = pts, sv="init_data.png")


    pts = fibonacci_sphere(360)
    phase_diffs = get_phase_diffs(pts, mic_locs, sample_freq)


    # PRINT FIRST PHASE DIFF
    # count = 0
    # for i in phase_diffs:
    #     if count == 1:
    #         break
    #     print(i, phase_diffs[i])
    #     count += 1

    # locs1 = lookup(data, phase_diffs, int(frames//2))
    # plot_with_mics(locs1, mic_locs_list, count = True, sv="peak1_lookup.png")


    locs2 = solve_implicit_function(data, int(frames//2), sample_freq, mic_locs_list)
    print(sum(locs2) / len(locs2))
    plot_with_mics(locs2, mic_locs_list, sv="all_tdoa.png")
    
    



    
    # for mic in tqdm.tqdm(pts):
    #     for peak in range(len(pts[mic])):
    #         pk1 = pts[mic][peak][0]

    #         pk1_min = pk1 - frames // 2
    #         pk1_max = pk1 + frames // 2

    #         new_mics = {}
    #         for n in mics:
    #             new_mics[n] = mics[n][int(pk1_min):int(pk1_max)]


    #         new_data = vstack_mics(new_mics, normalize = False)

    #         # plt_wavs(new_mics, sample_freq, pts = None, sv = "peak1.png")

            
    #         # pts = fibonacci_sphere(360)
    #         # phase_diffs = get_phase_diffs(pts, mic_locs, sample_freq)

    #         # locs1 = lookup(new_data, phase_diffs, int(frames//2))
    #         # plot_with_mics(locs1, mic_locs_list, count = True, sv="peak1_lookup.png")
            
    #         locs2 = solve_implicit_function(new_data, int(frames//2), sample_freq, mic_locs_list)
    #         print(locs2)
    #         print(sum(locs2) / len(locs2))
    #         localized_pts.append(sum(locs2) / len(locs2))

    # plot_with_mics(localized_pts, mic_locs_list, sv="peaks_tdoa.png")





    # SPERATION
    # W, A, final_data, source1, source2 = grad_descent_all_frames(data, frames)
    # print(source1.shape)
    # mic_to_wavs(source1, "./res2/src1")
    # mic_to_wavs(source2, "./res2/src2")
    # final_res_to_wavs(final_data, "./res2/")

    # folder = "./test/sound_clips/Michael_test1/1/"






    
    

    



    
    
 







