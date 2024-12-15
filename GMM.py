import numpy as np
import pywt
from sklearn.mixture import GaussianMixture

def wavelet_gmm(input_data, wavelet='db1', max_components=10):
    # Wavelet Decomposition
    coeffs = pywt.wavedec(input_data, wavelet)
    wavelet_coeffs = np.concatenate(coeffs)  # Flatten the coefficients

    # Fit a Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=max_components, covariance_type='full', random_state=69)
    gmm.fit(wavelet_coeffs.reshape(-1, 1))  # Reshape for sklearn

    # Step 3: Extract GMM parameters
    weights = gmm.weights_
    means = gmm.means_.flatten()
    variances = np.array([np.diag(cov) for cov in gmm.covariances_]).flatten() # Covariances are ignored

    return {
        'weights': weights,
        'means': means,
        'variances': variances
    }