import numpy as np
import pywt
from collections import Counter
from scipy import stats

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=stats.entropy(probabilities)
    return [entropy]
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)

    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    
    rms = np.nanmean(np.sqrt(list_values**2))

    mode, count = stats.mode(list_values)
    skew = stats.skew(list_values)
    min_value = np.min(list_values)
    max_value = np.max(list_values)
    
    return [n5, n25, n75, n95, median, mean, std, var, rms, mode[0], skew, min_value, max_value]
 
def calculate_crossings(list_values):
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_mean_crossings]

def calculate_wavelet(values, n_columns, waveletname):
    values = values.reshape((-1, n_columns))
    feature_vector = []
    
    for signal_comp in range(0,values.shape[1]):
        list_coeff = pywt.wavedec(values[:, signal_comp], waveletname)
        for coeff in list_coeff:
            entropy = calculate_entropy(coeff)
            crossings = calculate_crossings(coeff)
            statistics = calculate_statistics(coeff)
            feature_vector += entropy + crossings + statistics
    return feature_vector

def get_discrete_features(X, n_columns, waveletname, mains_mean=None, mains_std=None):
    
    X = np.array([calculate_wavelet(i, n_columns, waveletname) for i in X])
    
    if mains_mean is None:
        mains_mean = np.mean(X, axis=0)
        mains_std = np.std(X, axis=0)
        mains_std = np.where(mains_std < 1, 100, mains_std)
    
    X = (X - mains_mean) / mains_std

    return X, mains_mean, mains_std