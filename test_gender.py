import scipy.io
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import sys

male_filenames = {}
WIN_SHIFT = 25
WIN_SIZE = 50

import copy


def filename_to_wav(filename, plot_wav=False):
    path_to_wav = './wav_data/' + filename
    sample_rate, audio_buffer = scipy.io.wavfile.read(path_to_wav)

    if (plot_wav):
        duration = len(audio_buffer) / sample_rate

        time = np.arange(0, duration, 1 / sample_rate)  # time vector

        plt.plot(time, audio_buffer)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')

        label = 'Male' if filename in male_filenames else 'Female'

        plt.title('{}, {}'.format(filename, label))
        plt.show()

    return audio_buffer



def wav_to_frames(wav):
    num_frames = int(np.floor((wav.shape[0] - WIN_SIZE) / WIN_SHIFT) + 1)
    frames = np.zeros([WIN_SIZE, num_frames])
    for t in range(0, num_frames):
        frame = wav[t * WIN_SHIFT:t * WIN_SHIFT + WIN_SIZE]
        frame = frame - np.mean(frame)
        frames[:, t] = np.hamming(WIN_SIZE) * frame
    return frames.T


def get_zcr(framed_wav):
    n_frames, n_feat = framed_wav.shape
    zcr = np.zeros(n_frames)

    # я знаю что следующие три строчки стоило бы записать в одну, но могу так только в pandas
    for i in range(n_frames):
        for j in range(1, n_feat):
            zcr[i] = zcr[i] + np.sign(framed_wav[i][j - 1] * framed_wav[i][j])
    return np.mean(zcr), np.std(zcr)

#вот так, в лоб, я считаю амплитуду без преобразования фурье
def get_amplitude(sample):
    max_val = np.max(sample)
    min_val = np.min(sample)
    mean_val = np.mean(sample)
    return max(max_val - mean_val, mean_val - min_val)

def get_ste(framed_wav):
    amplitudes = np.array(list(map(get_amplitude, framed_wav)))
    return amplitudes.mean(), amplitudes.std()




def extract_features(framed_wav):
    result_table = pd.DataFrame(index=['zcr mean', 'zcr std',
                                       'ste mean', 'ste std'])
    zcr_m, zcr_s = get_zcr(framed_wav)
    ste_m, ste_s = get_ste(framed_wav)
    result_table[0] = [zcr_m, zcr_s, ste_m, ste_s]

    return result_table.T


source_table = pd.read_pickle("source_table.pkl")

def standartize_table(table, source_table=source_table):
    # table -- таблица которую хотим преобразовать
    # source_table -- таблица, на основе которой хотим преобразовать table. И для трейна, и для теста
    # этой таблицей будет служить train, ну классика короче

    for col in source_table.columns:
        table[col] = (table[col] - source_table[col].mean()) / source_table[col].std()
    return table

pca = pickle.load(open('pca.sav', 'rb'))
svm = pickle.load(open('svm.sav', 'rb'))

def main(filename):
    wav = filename_to_wav(filename)
    frames = wav_to_frames(wav)
    table = extract_features(frames)
    data = standartize_table(table)
    pca_data = pca.transform(data)
    prediction = svm.predict(pca_data)
    print('Male' if prediction == 1 else 'Female')

if __name__ == "__main__":
    main(sys.argv[1:][0])