import numpy as np
from scipy.signal import butter, lfilter

LAMBDA = 1
FILTER_ORDER = 2

def __high_frequency_content(signal, hop_size):
    N = int(signal.size / hop_size)
    hfc = np.zeros(N)
    for n in range(0, N):
        frame = signal[n * hop_size : (n+1) * hop_size]
        trans = np.fft.fft(frame)
        trans = trans[0:int(trans.size/2)]
        for i in range(0, trans.size):
            hfc[n] += i * trans[i].real
    return hfc

def __spectral_diff(signal, hop_size):
    N = int(signal.size / hop_size)
    sd = np.zeros(N)
    prev_trans = np.zeros(int(hop_size / 2))
    for n in range(0, N):
        frame = signal[n * hop_size : (n+1) * hop_size]
        trans = np.fft.fft(frame)
        trans = trans[0:int(trans.size/2)]
        for i in range(0, trans.size):
            if trans[i] < prev_trans[i]:
                continue
            sd[n] += (trans[i].real - prev_trans[i].real) ** 2
    return sd

def __phase_deviation(signal, hop_size):
    N = int(signal.size / hop_size)
    pd = np.zeros(N)
    mem1 = np.zeros(int(hop_size / 2))
    mem2 = np.zeros(int(hop_size / 2))
    for n in range(0, N):
        frame = signal[n * hop_size : (n+1) * hop_size]
        trans = np.fft.fft(frame)
        trans = trans[0:int(trans.size / 2)]
        pd[n] = sum([trans[k].imag - 2 * mem1[k].imag + mem2[k].imag for k 
            in range(0, trans.size)]) / trans.size
        mem2 = mem1
        mem1 = trans
    return pd

def __wavelet_regularity_modulus(signal, hop_size):
    N = int(signal.size / hop_size)
    return np.zeros(N)

def __negative_log_likelihood(signal, hop_size):
    N = int(signal.size / hop_size)
    return np.zeros(N)

def __reduction(signal, rt, hop_size):
    reduction_functions = {
            "hfc": __high_frequency_content,
            "sd": __spectral_diff,
            "pd": __phase_deviation,
            "wrm": __wavelet_regularity_modulus,
            "nll": __negative_log_likelihood
            }
    return reduction_functions[rt](signal, hop_size)

def __peak_detection(df, hop_size):
    onsets = []
    for i in range(1, df.size - 1):
        if df[i] > df[i-1] and df[i] > df[i+1] and df[i] > 0:
            onsets.append(((i-1) * hop_size, df[i]))
    return onsets

def __threshold(df, fs):
    M = int(0.1 * fs)
    D = max(abs(df)) / 2
    tf = np.zeros(df.size)
    for i in range(0, df.size):
        init = min([0, int(i - M / 2)])
        tf[i] = D + LAMBDA * np.median(df[init: i + int(M / 2)])
    return tf

def __postprocess(df):
    mean = np.mean(df)
    max_dev = max(abs(df - mean))
    if max_dev == 0:
        return df - mean
    return (df - mean) / max_dev

def __split_into_bands(signal, fs):
    bands = []
    lower = 44
    for i in range(3):
        upper = lower * 2
        bounds = (lower, upper)
        num, denom = butter(FILTER_ORDER, bounds, btype="bandpass", fs=fs)
        bands.append(lfilter(num, denom, signal))
        lower = upper
    for i in range(18):
        upper = lower * (2 ** (1/3))
        bounds = (lower, upper)
        try:
            num, denom = butter(FILTER_ORDER, bounds, btype="bandpass", fs=fs)
        except ValueError:
            continue
        bands.append(lfilter(num, denom, signal))
        lower = upper
    return bands

def __filter_band(onsets, fs):
    '''
    Drop any onset candidates that are within 50ms of a stronger candidate
    '''
    onsets = sorted(onsets, key = lambda o: o[0])
    window = 0.05 * fs
    new_onsets = []
    for i in range(len(onsets)):
        onset = onsets[i]
        prev_onset = onsets[i-1]
        if i+1 < len(onsets):
            next_onset = onsets[i+1]
        else:
            next_onset = (0,0)
        if abs(onset[0] - prev_onset[0]) < window:
            if prev_onset[1] > onset[1]:
                continue
        if abs(onset[0] - next_onset[0]) < window:
            if next_onset[1] > onset[1]:
                continue
        new_onsets.append(onset)
    return new_onsets

def __combine_bands(onset_candidates, fs, threshold):
    onset_candidates = sorted(onset_candidates, key = lambda o: o[0])
    window = 0.05 * fs
    new_onsets = []
    for o in onset_candidates:
        intensity = sum([on[1] for on in onset_candidates if abs(on[0] - o[0]) < window])
        if intensity > threshold:
            new_onsets.append((o[0], intensity))
    return new_onsets

def __convert_samples_to_samples(onsets, hop_size, fs):
    return onsets

def __convert_samples_to_frames(onsets, hop_size, fs):
    return onsets // hop_size

def __convert_samples_to_seconds(onsets, hop_size, fs):
    return onsets / fs

def detect_onsets(signal, fs, rt="hfc", hop_length=512, units="samples",
        split_bands=False, threshold=0):
    '''
    Onset detection function, utilising 5 different reduction functions.
    @param signal A numpy array representing the signal from which event onsets 
    are to be detected.
    @param rt Type of reduction function. Can be one of: hfc (high frequency
    content), sd (spectral difference), pd (phase deviation), wrm (wavelet
    regularity modulus) and nll (negative log likelihood).
    @param hop_length Length of frame in samples.
    @param units The units in which detected onset times are measured.
    Can be one of: frames, samples, time. Default is samples.
    @param split_bands Whether to perform analysis per-band or on the whole
    signal.
    @param threshold The threshold value of reduction function that a combined
    onset candidate has to have to be considered.
    @return A list of samples at which onsets have been detected.
    '''
    unit_conversions = {
            "samples": __convert_samples_to_samples,
            "frames": __convert_samples_to_frames,
            "time": __convert_samples_to_seconds
            }
    if split_bands:
        bands = __split_into_bands(signal, fs)
        onset_candidates = []
        for band in bands:
            detection_function = __postprocess(__reduction(band, rt, hop_length))
            threshold_function = __threshold(detection_function, fs)
            peaks = __peak_detection(detection_function - threshold_function, 
                    hop_length)
            peaks = __filter_band(peaks, fs)
            onset_candidates.extend(peaks)
        onsets = __combine_bands(onset_candidates, fs, threshold)
    else:
        detection_function = __reduction(signal, rt, hop_length)
        threshold_function = __threshold(detection_function, fs)
        onsets = __peak_detection(detection_function - threshold_function, 
                hop_length)
    onset_times = np.asarray([o[0] for o in onsets])
    return unit_conversions[units](onset_times, hop_length, fs)
