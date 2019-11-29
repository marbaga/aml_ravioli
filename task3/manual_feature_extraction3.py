import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import biosppy
import sys
import numpy
import time
from scipy.stats import skew, kurtosis
numpy.set_printoptions(threshold=sys.maxsize)
from scipy.fftpack import fft
import pywt
import scipy
from collections import Counter

def get_fft_values(y_values, T, N):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def extract_morphological_features(x): #extracts 22 temporal features
    features = []
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)

    # RR intervals
    rr = np.diff(rpeaks) * (1000 / 300)

    # fix for when biosppy fails to calculate hr
    if len(hr)==0:
        hr = 1000/rr
        print('Fix used')

    for index in [len(templates), np.mean(rr), np.median(rr), np.std(rr), calculate_entropy(rr), np.mean(hr), np.median(hr), np.std(hr), np.min(hr), np.max(hr)]:
        features.append(index)

    nn50 = np.sum(np.abs(np.diff(rr)) > 50)
    features.append(nn50)
    pnn50 = 100 * nn50 / len(rr)
    features.append(pnn50)
    nn20 = np.sum(np.abs(np.diff(rr)) > 20)
    features.append(nn20)
    pnn20 = 100 * nn20 / len(rr)
    features.append(pnn20)

    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    features.append(rmssd)
    sdsd = np.std(np.diff(rr))
    features.append(sdsd)

    rrv = np.diff(rr)
    rra = np.diff(rrv)
    for data in [rrv, rra]:
        for function in [np.mean, np.median, np.std]:
            features.append(function(data))

    return features

def extract_amplitude_features(row): #extracts 16 features on amplitude of the signal
    features = []
    for index in [np.max(row), np.min(row), np.mean(row), np.median(row), np.std(row), skew(row), kurtosis(row), calculate_entropy(row)]:
        features.append(index)

    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)
    avg_max_amp=0
    avg_min_amp=0
    avg_mean_amp=0
    avg_median_amp=0
    avg_var_amp=0
    avg_std_amp=0
    avg_skew=0
    avg_kurt=0
    for t in templates:
        avg_max_amp += np.max(t)
        avg_min_amp += np.min(t)
        avg_mean_amp += np.average(t)
        avg_median_amp += np.median(t)
        avg_std_amp += np.std(t)
        avg_var_amp = np.var(t)
        avg_skew = skew(t)
        avg_kurt = kurtosis(t)
    avg_max_amp/=len(templates)
    avg_min_amp/=len(templates)
    avg_mean_amp/=len(templates)
    avg_median_amp/=len(templates)
    avg_std_amp/=len(templates)
    avg_var_amp/=len(templates)
    avg_skew/=len(templates)
    avg_kurt/=len(templates)
    features.append(avg_max_amp)
    features.append(avg_min_amp)
    features.append(avg_mean_amp)
    features.append(avg_median_amp)
    features.append(avg_std_amp)
    features.append(avg_var_amp)
    features.append(avg_skew)
    features.append(avg_kurt)
    return features

def extract_fourier_features(x): #returns the frequencies fo the 5 highest peaks AND the peak value in each 0.25 interval from 0 to 5 Hz, also average values for 3 significant bands and the median frequency (29 features)
    f_values, fft_values = get_fft_values(x, 1/300, len(x))
    ind = np.argpartition(fft_values, -5)[-5:]
    features = np.zeros((26))
    LF=[]
    MF=[]
    HF=[]
    for i in range(0, len(f_values)):
        if int(f_values[i]/0.25) > 19:
            break
        if fft_values[i] >= features[int(f_values[i]/0.25)]:
            features[int(f_values[i]/0.25)]=fft_values[i]
        if f_values[i] <=0.04 and f_values[i] > 0.0033:
            LF.append(fft_values[i])
        if f_values[i] <=0.15 and f_values[i] >0.04:
            MF.append(fft_values[i])
        if f_values[i] <=0.4 and f_values[i] >0.15:
            HF.append(fft_values[i])

    curr = 20
    for f in f_values[ind]:
        features[curr]=f
        curr = curr + 1
    features[curr]=np.sum(f_values*fft_values)/np.sum(fft_values)

    if len(LF) == 0:
        LF.append(0)
    if len(MF) == 0:
        MF.append(0)
    if len(HF) == 0:
        HF.append(0)

    return list(features) + [np.mean(LF), np.mean(MF), np.mean(HF)]

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

def get_percentiles(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    return [n5, n25, n75, n95]

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
    #return [mean, std, rms]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + statistics

def get_wavelet_features(x, waveletname='db4'):
    list_coeff = pywt.wavedec(x, waveletname, level=4)[4]
    return get_features(list_coeff)

def extract_qrs_features(x): #should extract features from pqrst complexes
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)

    q_points = []
    s_points = []
    for peak in rpeaks:
        i = peak
        while filtered[i] > filtered[i - 1]:
            i -= 1
        q_points.append(i)
        i = peak + 1
        while filtered[i] > filtered[i + 1]:
            i += 1
        s_points.append(i)

    qs_intervals = []
    for i in range(max(len(q_points), len(s_points))):
        qs_intervals.append(s_points[i] - q_points[i])

    q_amplitudes = []
    s_amplitudes = []
    for i in q_points:
        q_amplitudes.append(filtered[i])
    for i in s_points:
        s_amplitudes.append(filtered[i])

    features = []
    features.append(np.average(qs_intervals))  # mean of qs intervals length
    features.append(np.std(qs_intervals))  # std of qs intervals length
    features.append(np.average(q_amplitudes))
    features.append(np.std(q_amplitudes))
    features.append(np.average(s_amplitudes))
    features.append(np.std(s_amplitudes))

    return features

print('Starting')
X_t = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
print('Read')
print(X_t.shape)

X=[]
for i, row in enumerate(X_t):
    x = row[np.logical_not(np.isnan(row))]
    try:
        features=[]
        features += extract_morphological_features(x)
        features += extract_amplitude_features(x)
        features += extract_fourier_features(x)
        features += get_wavelet_features(x)
        features += extract_qrs_features(x)
        X.append(features)
        print(features)
        print(str(i) + '/' + str(X_t.shape[0]) + ' with ' + str(len(features)) + ' features')
    except:
        X.append([0])
        print('ERROR ' + str(i) + '/' + str(X_t.shape[0]))


pd.DataFrame(X).to_csv('X_test_manual_extraction9.csv')
#X = pd.read_csv('X_test_manual_extraction9.csv', ',').iloc[:, 1:].to_numpy()
print('Features extracted')

'''
        f_values, fft_values = get_fft_values(x, 1/300, len(x), 300)
        plt.plot(f_values, fft_values, linestyle='-', color='blue')
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()
'''