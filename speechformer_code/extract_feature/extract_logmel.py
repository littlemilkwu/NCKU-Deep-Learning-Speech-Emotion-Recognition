
import soundfile 
import librosa
from scipy import signal
import numpy as np
import pandas as pd
from scipy import io
import os
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def extract_logmel(wavfile: str, savefile: str, window_size: int, hop: int, ham_window, filter_bank):
    '''
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    # x, fs = soundfile.read(wavfile)
    x, fs = librosa.load(wavfile)
    
    if x.ndim > 1:
        x = np.mean(x, axis=-1)

    f, t, x = signal.spectral.spectrogram(x, fs, window=ham_window, nperseg=window_size, noverlap=window_size - hop, detrend=False, mode='magnitude')
    x = np.dot(x.T, filter_bank.T).T   # Hz -> mel
    x = np.log(x + 1e-8)     # mel -> log_mel

    x = x.T   # -> (t, d)
    # print(savefile, x.shape)

    dict = {'logmel': x}
    io.savemat(savefile, dict)

if __name__ == '__main__':
    frame = 0.025   # second
    hop = 0.01      # second
    sr = 44100      # sample rate  16000 (iemocap, daiz_woc) or 44100 (meld, pitt)
    window_size = int(sr * frame)
    hop = int(sr * hop)
    n_mels = 128
    ham_window = np.hamming(window_size)
    filter_bank = librosa.filters.mel(sr=sr, n_fft=window_size, n_mels=n_mels)

    #### use extract_logmel
    path = "/home/chy1024/SER/SpeechFormer/DL_data/"
    train = pd.read_csv("/home/chy1024/SER/SpeechFormer/DL_data/train_data.csv")
    # file_direct_list = os.listdir(path)
    bar = tqdm(range(len(train)), position=0, leave=True)
    for i in bar:
        wavfile = path + train.iloc[i, 0] + "/" + train.iloc[i, 1]
        filename = train.iloc[i, 1].split(".")[0]
        savefile = f"/home/chy1024/SER/SpeechFormer/DL_data/logmel_25ms_mat/train/{filename}.mat"
        # if os.path.isfile(savefile):
        #     continue
        # else:
        extract_logmel(wavfile, savefile, window_size, hop, ham_window, filter_bank)
        bar.set_postfix(savefile=savefile.split('/')[-1])
        
    