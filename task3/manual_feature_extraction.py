import pandas as pd
import numpy as np
import neurokit as nk
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

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


'''
Features to consider:

amplitude of q,r,s peaks (hard to measure)

interval between the current and previous beat and the one between the current and subsequent beat, which are called RR1 and RR2 respectively. Another interval is defined as the distance between the previous beat and its predecessor, called RR0.
ratios: Ratio1=RR0/RR1 Ratio2=RR2/RR1 Ratio3=RRm/RR1 where RRm=mean(RR0,RR1,RR2)

try fourier transform?
- v1: find frequencies at which coefficients vary the most along different samples. each sample is charachterized by the coefficients at that frequency
- v3: find the set of frequencies at which the highest peaks occur, then same thing
- v3: 5 features for each samples are the frequencies at which they peak 
'''

def extract_morphological_features(row):
    features = []
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)
    # RR intervals
    rr = np.diff(rpeaks) * (1000 / 300)
    #features.append(rr)
    #print('RR intervals: ' + str(rr))

    # fix for when biosppy fails to calculate hr
    if len(hr)==0:
        hr = 1000/rr
        print('Fix used')

    # Number of pulses
    rr_len = len(rr)
    features.append(rr_len)
    #print('Number of beats: ', rr_len)
    # Mean RR
    mean_rr = np.mean(rr)
    features.append(mean_rr)
    #print('Mean RR: ' + str(mean_rr))
    # Standard deviation RR
    sdrr = np.std(rr)
    features.append(sdrr)
    #print('Std RR: ' + str(sdrr))
    # RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    features.append(rmssd)
    #print('RMSSD: ' + str(rmssd))
    # Mean HR
    mean_hr = np.mean(hr)
    features.append(mean_hr)
    #print('Mean HR: ' + str(mean_hr))
    # STD HR
    std_hr = np.std(hr)
    features.append(std_hr)
    #print('Std HR: ' + str(std_hr))
    # Min HR
    min_hr = np.min(hr)
    features.append(min_hr)
    #print('Min HR: ' + str(min_hr))
    # Max HR
    max_hr = np.max(hr)
    features.append(max_hr)
    #print('Max HR: ' + str(max_hr))
    # NNxx: sum absolute differences that are larger than 50ms
    nnxx = np.sum(np.abs(np.diff(rr)) > 50)
    features.append(nnxx)
    #print('NNXX: ' + str(nnxx))
    # pNNx: fraction of nnxx of all rr-intervals
    pnnx = 100 * nnxx / len(rr)
    features.append(pnnx)
    #print('PNNX: ' + str(pnnx))
    return features

def extract_amplitude_features(row):
    features=[]
    features = []
    max_amp = np.max(row)
    features.append(max_amp)
    min_amp = np.min(row)
    features.append(min_amp)
    mean_amp = np.mean(row)
    features.append(mean_amp)
    std_amp = np.std(row)
    features.append(std_amp)
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)
    avg_max_amp=0
    avg_min_amp=0
    avg_mean_amp=0
    avg_std_amp=0
    avg_skew=0
    avg_kurt=0
    for t in templates:
        avg_max_amp += np.max(t)
        avg_min_amp += np.min(t)
        avg_mean_amp += np.average(t)
        avg_std_amp += np.std(t)
        avg_var = np.var(t)
        avg_skew = skew(t)
        avg_kurt = kurtosis(t)
    avg_max_amp/=len(templates)
    avg_min_amp/=len(templates)
    avg_mean_amp/=len(templates)
    avg_std_amp/=len(templates)
    avg_skew/=len(templates)
    avg_kurt/=len(templates)
    features.append(avg_max_amp)
    features.append(avg_min_amp)
    features.append(avg_mean_amp)
    features.append(avg_std_amp)
    features.append(avg_skew)
    features.append(avg_kurt)
    return features

def extract_fourier_features(x): #return the frequencies fo the 5 highest peaks AND the peak value in each 0.25 interval from 0 to 5 Hz
    f_values, fft_values = get_fft_values(x, 1/300, len(x), 300) #here change len(x) with something else
    ind = np.argpartition(fft_values, -5)[-5:]
    features = np.zeros((26))
    for i in range(0, len(f_values)):
        if int(f_values[i]/0.25) > 19:
            break
        if fft_values[i] >= features[int(f_values[i]/0.25)]:
            features[int(f_values[i]/0.25)]=fft_values[i]
    curr = 20
    for f in f_values[ind]:
        features[curr]=f
        curr = curr + 1
    features[curr]=np.sum(f_values*fft_values)/np.sum(fft_values)
    return list(features)

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

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

print('Starting')
X_t = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
#X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
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
        X.append(features)
        print(str(i) + '/' + str(X_t.shape[0]) + ' with ' + str(len(features)) + ' features')
    except:
        X.append([0])
        print('ERROR ' + str(i) + '/' + str(X_t.shape[0]))

pd.DataFrame(X).to_csv('X_test_manual_extraction.csv')
X = pd.read_csv('X_test_manual_extraction.csv', ',').iloc[:, 1:].to_numpy()
print('Features extracted')

'''
        f_values, fft_values = get_fft_values(x, 1/300, len(x), 300)
        plt.plot(f_values, fft_values, linestyle='-', color='blue')
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()
'''


























'''
import pandas as pd
import numpy as np
import neurokit as nk
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

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values



def extract_morphological_features(row):

    features = []
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)
    # RR intervals
    rr = np.diff(rpeaks) * (1000 / 300)
    #features.append(rr)
    #print('RR intervals: ' + str(rr))
    # Number of pulses
    rr_len = len(rr)
    features.append(rr_len)
    #print('Number of beats: ', rr_len)
    # Mean RR
    mean_rr = np.mean(rr)
    features.append(mean_rr)
    #print('Mean RR: ' + str(mean_rr))
    # Standard deviation RR
    sdrr = np.std(rr)
    features.append(sdrr)
    #print('Std RR: ' + str(sdrr))
    # RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    features.append(rmssd)
    #print('RMSSD: ' + str(rmssd))
    # Mean HR
    mean_hr = np.mean(hr)
    features.append(mean_hr)
    #print('Mean HR: ' + str(mean_hr))
    # STD HR
    std_hr = np.std(hr)
    features.append(std_hr)
    #print('Std HR: ' + str(std_hr))
    # Min HR
    min_hr = np.min(hr)
    features.append(min_hr)
    #print('Min HR: ' + str(min_hr))
    # Max HR
    max_hr = np.max(hr)
    features.append(max_hr)
    #print('Max HR: ' + str(max_hr))
    # NNxx: sum absolute differences that are larger than 50ms
    nnxx = np.sum(np.abs(np.diff(rr)) > 50)
    features.append(nnxx)
    #print('NNXX: ' + str(nnxx))
    # pNNx: fraction of nnxx of all rr-intervals
    pnnx = 100 * nnxx / len(rr)
    features.append(pnnx)
    #print('PNNX: ' + str(pnnx))
    return features

def extract_amplitude_features(row):
    features=[]
    features = []
    max_amp = np.max(row)
    features.append(max_amp)
    min_amp = np.min(row)
    features.append(min_amp)
    mean_amp = np.mean(row)
    features.append(mean_amp)
    std_amp = np.std(row)
    features.append(std_amp)
    ts, filtered, rpeaks, templates_ts, templates, hr_ts, hr = biosppy.signals.ecg.ecg(x, sampling_rate=300, show=False)
    avg_max_amp=0
    avg_min_amp=0
    avg_mean_amp=0
    avg_std_amp=0
    avg_skew=0
    avg_kurt=0
    for t in templates:
        avg_max_amp += np.max(t)
        avg_min_amp += np.min(t)
        avg_mean_amp += np.average(t)
        avg_std_amp += np.std(t)
        avg_var = np.var(t)
        avg_skew = skew(t)
        avg_kurt = kurtosis(t)
    avg_max_amp/=len(templates)
    avg_min_amp/=len(templates)
    avg_mean_amp/=len(templates)
    avg_std_amp/=len(templates)
    avg_skew/=len(templates)
    avg_kurt/=len(templates)
    features.append(avg_max_amp)
    features.append(avg_min_amp)
    features.append(avg_mean_amp)
    features.append(avg_std_amp)
    features.append(avg_skew)
    features.append(avg_kurt)
    return features

def extract_fourier_features(x): #return the frequencies fo the 5 highest peaks AND the peak value in each 0.25 interval from 0 to 5 Hz
    f_values, fft_values = get_fft_values(x, 1/300, len(x), 300) #here change len(x) with something else
    ind = np.argpartition(fft_values, -5)[-5:]
    features = np.zeros((26))
    for i in range(0, len(f_values)):
        if int(f_values[i]/0.25) > 19:
            break
        if fft_values[i] >= features[int(f_values[i]/0.25)]:
            features[int(f_values[i]/0.25)]=fft_values[i]
    curr = 20
    for f in f_values[ind]:
        features[curr]=f
        curr = curr + 1
    features[curr]=np.sum(f_values*fft_values)/np.sum(fft_values)
    return list(features)

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

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

print('Starting')
X_t = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
#X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()
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
        X.append(features)
        print(str(i) + '/' + str(X_t.shape[0]))
    except:
        X.append([0])
        print('ERROR ' + str(i) + '/' + str(X_t.shape[0]))

#pd.DataFrame(X).to_csv('extracted_features.csv')
X = pd.read_csv('extracted_features.csv', ',').iloc[:, 1:].to_numpy()
print('Features extracted')


        f_values, fft_values = get_fft_values(x, 1/300, len(x), 300)
        plt.plot(f_values, fft_values, linestyle='-', color='blue')
        plt.xlabel('Frequency [Hz]', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.show()
'''