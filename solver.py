import numpy as np
import pywt
from scipy.stats import norm


def initialize_noise_variance(input_data):
    T = len(input_data)
    noise_variance = np.zeros(T)

    for t in range(T):
        prev_diff = (input_data[t] - input_data[t - 1])**2 if t > 0 else (input_data[t] - input_data[T - 1])**2
        next_diff = (input_data[t] - input_data[t + 1])**2 if t < T - 1 else (input_data[t] - input_data[0])**2
        noise_variance[t] = 0.5 * (prev_diff + next_diff)

    coeffs = pywt.wavedec(noise_variance, 'db1')
    # coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    res = np.array([np.sum(c**2) for c in coeffs])
    return res # coeff_arr, coeff_slices


def update_noise_variance(y, mu, noise_variance, wavelet='db1', threshold_scaling=1.0):
    # Compute squared residuals
    Z_squared = (y - mu) ** 2

    # Apply discrete wavelet transform (DWT)
    coeffs = pywt.wavedec(Z_squared, wavelet)

    # Soft thresholding
    # Compute threshold for each detail coefficient
    thresholded_coeffs = []
    n = len(y)
    for i, coeff in enumerate(coeffs):
        if i == 0:  # Skip approximation coefficients
            thresholded_coeffs.append(coeff)
            continue

        sigma_hat = np.std(coeff)  # Estimate noise level
        threshold = threshold_scaling * np.sqrt(2 * np.log(n)) * sigma_hat
        coeff_shrunk = pywt.threshold(coeff, threshold, mode='soft')
        thresholded_coeffs.append(coeff_shrunk)

    # Reconstruct the variance function using inverse DWT
    variance_estimate = thresholded_coeffs # pywt.waverec(thresholded_coeffs, wavelet)
    res = np.array([np.sum(c**2) for c in thresholded_coeffs])
    return res


def shrink_wavelet_coefficients(wavelet_coeffs, gmm_params, noise_variances):
    weights = np.array(gmm_params['weights'])  # pi_k
    means = np.array(gmm_params['means'])      # m_k
    variances = np.array(gmm_params['variances'])  # tau_k^2

    # Initialize the array for shrunk coefficients
    shrunk_coeffs = np.zeros_like(wavelet_coeffs)

    # Loop over each wavelet coefficient
    for j, y_j in enumerate(wavelet_coeffs):
        print(j)
        print(y_j)
        omega_j2 = noise_variances[j]  # Noise variance for this coefficient

        # Initialize posterior weights and means for this coefficient
        posterior_weights = np.zeros_like(weights)
        posterior_means = np.zeros_like(weights)

        # Compute posterior weights and means for each mixture component
        for k in range(len(weights)):
            tau_k2 = variances[k]
            m_k = means[k]

            # Posterior variance and mean
            v_jk = 1 / (1 / omega_j2 + 1 / tau_k2)  # Posterior variance
            mu_jk = v_jk * (y_j / omega_j2 + m_k / tau_k2)  # Posterior mean

            # Store posterior mean for this component
            posterior_means[k] = mu_jk

            # Compute likelihood
            likelihood = norm.pdf(y_j, loc=m_k, scale=np.sqrt(tau_k2 + omega_j2))
            posterior_weights[k] = weights[k] * likelihood

        # Normalize posterior weights
        posterior_weights /= np.sum(posterior_weights)

        # Compute the weighted posterior mean
        shrunk_coeffs[j] = np.sum(posterior_weights * posterior_means)

    return shrunk_coeffs


# def shrink_wavelet_coefficients(wavelet_coeffs, gmm_params, noise_variance):
#     """
#     Shrinks wavelet coefficients using posterior means.

#     Parameters:
#         wavelet_coeffs (array-like): Noisy wavelet coefficients (y_t).
#         gmm_params (dict): GMM parameters with keys:
#             - 'weights': Weights of the Gaussian components (pi_k).
#             - 'means': Means of the Gaussian components (m_k).
#             - 'variances': Variances of the Gaussian components (v_k).
#         noise_variance (float): Noise variance (sigma^2).

#     Returns:
#         array-like: Shrunk wavelet coefficients (mu_t).
#     """
#     weights = gmm_params['weights']
#     means = gmm_params['means']
#     variances = gmm_params['variances']

#     # Initialize the array for shrunk coefficients
#     shrunk_coeffs = np.zeros_like(wavelet_coeffs)

#     for i, y in enumerate(wavelet_coeffs):
#         # Compute posterior probabilities w_k(y)
#         posterior_weights = np.zeros_like(weights)
#         posterior_means = np.zeros_like(weights)

#         for k in range(len(weights)):
#             v_k = variances[k]
#             m_k = means[k]

#             posterior_means[k] = (noise_variance * m_k + v_k * y) / (noise_variance + v_k)
#             likelihood = norm.pdf(y, loc=m_k, scale=np.sqrt(v_k + noise_variance))
#             posterior_weights[k] = weights[k] * likelihood

#         # Normalize posterior weights
#         posterior_weights /= np.sum(posterior_weights)

#         # Compute the weighted posterior mean
#         shrunk_coeffs[i] = np.sum(posterior_weights * posterior_means)

#     return shrunk_coeffs


def iterative_noise_variance_estimation(input_data, wavelet, gmm_params, max_iter=10, tol=1e-6):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(input_data, wavelet)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    # Initialize noise variance
    noise_variance = initialize_noise_variance(input_data)

    for iteration in range(max_iter):
        # Shrink wavelet coefficients using current noise variance
        mu = shrink_wavelet_coefficients(coeff_arr, gmm_params, noise_variance)

        # Update noise variance
        updated_variance = update_noise_variance(coeff_arr, pywt.waverec(mu, wavelet), noise_variance)

        # Check for convergence
        if np.abs(updated_variance - noise_variance) < tol:
            break

        noise_variance = updated_variance

    # Reconstruct the denoised signal
    shrunk_coeffs = pywt.array_to_coeffs(mu, coeff_slices, output_format='wavedec')
    denoised_signal = pywt.waverec(shrunk_coeffs, wavelet)

    return noise_variance, denoised_signal
