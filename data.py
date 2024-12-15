import numpy as np
import matplotlib.pyplot as plt

# Define the blip function
def blip_fm(t, signal_type="mean"):
    fn = (0.12 + 3.6 * t + 0.3 * np.exp(-100 * (t - 0.3)**2)) * ((t >= 0) & (t <= 0.8)) + \
         (-0.28 + 0.6 * t + 0.3 * np.exp(-100 * (t - 1.3)**2)) * ((t > 0.8) & (t <= 1))
    if signal_type == "mean":
        return fn
    elif signal_type == "var":
        return np.zeros_like(fn)

# Define the corner function
def cor_fm(t, signal_type="mean"):
    fn = 623.87 * t**3 + (1 - 2 * t) * ((t >= 0) & (t <= 0.5)) + 187.161 * (0.125 - t**3) * ((t > 0.5) & (t <= 0.8)) + \
         3788.470441 * (t - 1)**3 * ((t > 0.8) & (t <= 1))
    if signal_type == "mean":
        return fn
    elif signal_type == "var":
        return np.zeros_like(fn)

# Define the cornered blocks function
def cblocks_fm(t, signal_type="mean"):
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = 2.88 / 5 * np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    fn = np.zeros_like(t)
    for i in range(len(pos)):
        fn += (1 + np.sign(t - pos[i])) * (hgt[i] / 2)
    if signal_type == "mean":
        return fn
    elif signal_type == "var":
        return 1e-5 + 1 * (fn - np.min(fn)) / np.max(fn)

# Define the temporal exponential function
def texp_fm(t, signal_type="mean"):
    fn = 1e-4 * 4 * (np.exp(-550 * (t - 0.2)**2) + np.exp(-200 * (t - 0.5)**2) + np.exp(-950 * (t - 0.8)**2))
    if signal_type == "mean":
        return fn
    elif signal_type == "var":
        return np.zeros_like(fn)

# Define the constant function
def cons_fm(t, signal_type="mean"):
    fn = np.ones_like(t)
    if signal_type == "mean":
        return fn
    elif signal_type == "var":
        return fn

# Gaussian signal generator
def gaussian_id(n=100, rsnr=1, meanfn=blip_fm, varfn=None):
    t = np.linspace(0, 1, n)
    mu = meanfn(t, signal_type="mean")
    sigma = np.sqrt(varfn(t, signal_type="var") if varfn else np.zeros_like(t))
    sig_true = sigma / np.mean(sigma) * np.std(mu) / rsnr**2
    x_data = np.random.normal(mu, sig_true)
    sig_est = np.sqrt((2 / 3) * np.sum(np.diff(x_data, 2)**2))
    return t, mu, x_data, sig_true, sig_est

# Define the signal generation functions
def spikes(t):
    spike_f = (
        0.75 * np.exp(-500 * (t - 0.23)**2)
        + 1.5 * np.exp(-2000 * (t - 0.33)**2)
        + 3 * np.exp(-8000 * (t - 0.47)**2)
        + 2.25 * np.exp(-16000 * (t - 0.69)**2)
        + 0.5 * np.exp(-32000 * (t - 0.83)**2)
    )
    return (1 + spike_f) / 5

def bumps(t):
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = 2.97 / 5 * np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array([0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005])
    fn = np.zeros_like(t)
    for p, h, w in zip(pos, hgt, wth):
        fn += h / ((1 + (np.abs(t - p) / w))**4)
    return (1 + fn) / 5

def blocks(t):
    pos = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = 2.88 / 5 * np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    fn = np.zeros_like(t)
    for p, h in zip(pos, hgt):
        fn += (1 + np.sign(t - p)) * (h / 2)
    fn = 0.2 + 0.6 * (fn - fn.min()) / (fn.max() - fn.min())
    return fn

def angles(t):
    sig = (
        (2 * t + 0.5) * (t <= 0.15)
        + (-12 * (t - 0.15) + 0.8) * (t > 0.15) * (t <= 0.2)
        + 0.2 * (t > 0.2) * (t <= 0.5)
        + (6 * (t - 0.5) + 0.2) * (t > 0.5) * (t <= 0.6)
        + (-10 * (t - 0.6) + 0.8) * (t > 0.6) * (t <= 0.65)
        + (-0.5 * (t - 0.65) + 0.3) * (t > 0.65) * (t <= 0.85)
        + (2 * (t - 0.85) + 0.2) * (t > 0.85)
    )
    sig = 3 / 5 * ((5 / (sig.max() - sig.min())) * sig - 1.6) - 0.0419569
    return (1 + sig) / 5

def doppler(t):
    doppler_f = np.sqrt(t * (1 - t)) * np.sin((2 * np.pi * 1.05) / (t + 0.05))
    doppler_f = 3 / (doppler_f.max() - doppler_f.min()) * (doppler_f - doppler_f.min())
    return (1 + doppler_f) / 5
