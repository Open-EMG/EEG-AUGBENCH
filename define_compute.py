
import numpy as np
from pywt import wavedec


def extract_features_all(X, sfreq=256):
    epochs, channel, sample_length = X.shape
    features = np.zeros((epochs, 3*channel*4))
    for i in range(epochs):
        delta, theta, alpha, beta = wavedec(X[i,0,:], 'db1', level=3)
        temp_compute_energy_dwt = compute_energy_dwt(delta, theta, alpha, beta)
        temp_compute_variance_dwt = compute_variance_dwt(delta, theta, alpha, beta)
        temp_compute_TK_energy_dwt = compute_TK_energy_dwt(delta, theta, alpha, beta)
       
        features[i, :] = np.concatenate((temp_compute_variance_dwt, temp_compute_energy_dwt, temp_compute_TK_energy_dwt),axis=0)
    return features

def compute_energy_dwt(delta, theta, alpha, beta):
    features = [compute_energy(i) for i in (delta, theta, alpha, beta)]
    return np.concatenate(features)
def compute_energy(data):
    return np.sum(data**2, axis=-1)


def compute_variance_dwt(delta, theta, alpha, beta):
    features = [np.var(i, axis=-1, ddof=1) for i in (delta, theta, alpha, beta)]
    return np.concatenate(features)

def _tk_energy(data):
    n_channels, n_times = data.shape
    tke = np.empty((n_channels, n_times - 2), dtype=data.dtype)
    for j in range(n_channels):
        for i in range(1, n_times - 1):
            tke[j, i - 1] = data[j, i] ** 2 - data[j, i - 1] * data[j, i + 1]
    return tke


def compute_TK_energy_dwt(delta, theta, alpha, beta):
    """Teager-Kaiser Energy.
    """
    features = [np.mean(_tk_energy(i),axis=-1) for i in (delta, theta, alpha, beta)]
    return np.concatenate(features)
