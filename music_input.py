import numpy as np
from scipy import signal as sig
import librosa
import math
from matplotlib import pyplot as plt
import operator
import mido

import onset_detection
from typesystem import *

"""Size of analysis frame in seconds"""
FRAME_SIZE = 0.093
"""Minimum frequency of interest"""
FMIN = 40
"""Maximum frequency of interest"""
FMAX = 2100
"""The threshold silence salience- a frame with a salience lower than that is
considered silent"""
SILENCE_THRESHOLD = 10000
"""Period precision"""
TPREC = 1.5e-8
"""Minimum increase in the significance of the next F0 for it to be considered
significant"""
SIGINC = 1.1

def __filter_center_fcies():
    '''
    Computes a list of central frequencies for the bank of band-pass filters used
    to split the input signal into channels. 
    @return The list of frequencies containing 70 frequencies between roughly 64H
    and 5077Hz, all in Hz.
    '''
    filter_center_fcies = []
    e0 = 2.3
    e1 = 0.39
    no_of_filters = 70
    for i in range(0, no_of_filters):
        filter_center_fcies.append(229 * (10 ** ((i*e1+e0)/21.4) - 1))
    return filter_center_fcies

def __filter_bandwidths(center_fcies):
    '''
    Computes a list of bandwidths for the filter bank. 
    @param center_fcies A list of central frequencies of the filters.
    @return A list of filter bandwidths, equal in size to the list of central
    frequencies. All bandwidths are in Hz.
    '''
    bandwidths = []
    for f in center_fcies:
        bandwidths.append(0.108 * f + 24.7)
    return bandwidths

def __coefficients(fc, fs, bc):
    '''
    Computes the lists of coefficients for the first and the second filter types.
    @param fc The central frequency of the filter, in Hz.
    @param fs The sampling frequency of the signal, in Hz.
    @param bc The bandwidth of the filter, in Hz.
    @return A 4-tuple of four lists: numerator coefficients of the first filter,
    denomenator coefficients of the first filter, numerator coefficients of the
    second filter, denomenator coefficients of the second filter.
    '''
    angular_fcy = 2 * math.pi * fc / fs
    B = 1.019 * bc
    b3dB = 2 * B * math.sqrt(2 ** 0.25 - 1)
    A = math.exp((-b3dB*math.pi)/(fs*math.sqrt(2 ** 0.25 - 1)))
    theta1 = math.acos(((1+A**2)*math.cos(angular_fcy))/(2*A))
    theta2 = math.acos((2*A*math.cos(angular_fcy))/(1+A**2))
    p1 = 0.5 * (1 - A ** 2)
    p2 = (1-A) * math.sqrt(1 - math.cos(theta2)**2)

    num_coefs_1 = [p1, 0, -p1]
    denom_coefs_1 = [1, -A*math.cos(theta1), A**2]
    num_coefs_2 = [p2]
    denom_coefs_2 = [1, -A*math.cos(theta2), A**2]
    return num_coefs_1, denom_coefs_1, num_coefs_2, denom_coefs_2

def __dynamic_compression(signal, fs):
    '''
    Applies dynamic compression to the signal.
    @param signal The signal to be compressed, as a numpy array.
    @param fs Sampling frequency in Hz.
    @return A new numpy array with the signal dynamically compressed.
    '''
    # frames of 46ms
    samples_per_frame = int(fs * 0.046)
    pos = 0
    result = np.copy(signal)
    while pos + samples_per_frame <= signal.size:
        frame = signal[pos:pos + samples_per_frame]
        # standard deviation of the frame
        s = np.std(frame)
        # if the standard deviation is zero, the signal was zero all through the
        # frame, so there is nothing to be compressed
        if s == 0:
            pos += samples_per_frame
            continue
        sf = s ** (-0.66)
        result[pos:pos + samples_per_frame] = frame * sf
        pos += samples_per_frame
    return result

def __fwr_and_filter(signal, fs, fc):
    '''
    Applies full wave rectification and a low-pass filter.
    @param signal The signal to be processed, as a numpy array.
    @param fs Sampling frequency, in Hz.
    @param fc The cutoff frequency for the filter.
    @return A new numpy array with the processed signal.
    '''
    # design the lowpass filter
    fnum, fdenom = sig.butter(2, fc, fs=fs)
    # the abs function is effectively full wave rectification
    filtered = sig.lfilter(fnum, fdenom, abs(signal))
    return (filtered + signal) / 2

def __hamming_window(N):
    '''
    Constructs a Hamming window of given size.
    @param N the size of desired window, in samples.
    @return A numpy array representing the Hamming window.
    '''
    a = 25/46
    window = np.zeros(N)
    for i in range(0, N):
        window[i] = a - (1-a) * math.cos(2*math.pi/(N-1) * i)
    return window

def __weight(tau_low, tau_up, m, fs):
    '''
    Calculates the weight of a given partial in salience.
    @param tau_low The lowest possible period of the partial (in seconds).
    @param tau_up The highest possible period of the partial (in seconds).
    @param m The index of the partial.
    @param fs The sampling frequency (in Hz).
    @return The weight of the partial.
    '''
    # TODO: implement optimisation
    e1 = 20 # Hz
    e2 = 320 # Hz
    return (fs/tau_low+e1)/(m*fs/tau_up+e2)

def __salience(frame, fs, tau_low, tau_up):
    '''
    Calculates the salience (strength) of a period within the given period delta 
    in a frame.
    @param frame A numpy array representing the frame of signal in the frequency
    domain.
    @param fs The sampling frequency, in Hz.
    @param tau_low The minimum candidate period in seconds.
    @param tau_up The maximum candidate period in seconds.
    @return The salience of oscillation with the given period.
    '''
    # needed for conversion between frequency in Hz and frequency bins
    bin_width = fs / frame.size

    salience = 0
    for i in range(1,21):
        fmin = i / tau_up
        fmax = i / tau_low
        binmin = int(fmin / bin_width)
        binmax = int(fmax / bin_width)
        max_energy = 0
        for k in range(binmin, binmax+1):
            if k >= frame.size:
                continue
            energy = frame[k]
            if energy > max_energy:
                max_energy = energy
        salience += __weight(tau_low, tau_up, i, fs) * max_energy
    return salience

def __max_salience(summary_spectrum, fs):
    '''
    Calculates the period of oscillation with the maximum salience. This is done
    with effectively a binary search- the period range in question gets split in
    half and the half with most salience is chosen to recur.
    @param summary_spectrum A numpy array representing the frame of signal in the
    frequency domain.
    @param fs The sampling frequency in Hz.
    @return The period of highest salience in seconds and the corresponding 
    salience.
    '''
    Q = 0
    qbest = 0
    # period correlating to 2100Hz is the smallest
    tau_low = [1/FMAX]
    # period correlating to 40Hz is the largerst
    tau_up = [1/FMIN]
    saliences = [0]
    while tau_up[qbest] - tau_low[qbest] > TPREC:
        # split the best block and compute new limits
        Q += 1
        tau_low.append((tau_low[qbest] + tau_up[qbest])/2)
        tau_up.append(tau_up[qbest])
        tau_up[qbest] = tau_low[Q]
        # compute saliences for the two halves
        saliences[qbest] = __salience(summary_spectrum, fs, tau_low[qbest], 
                tau_up[qbest])
        saliences.append(__salience(summary_spectrum, fs, tau_low[Q], tau_up[Q]))
        # choose the block with the highest salience to split further
        qbest, _ = max(enumerate(saliences), key=operator.itemgetter(1))
    tau_prime = (tau_low[qbest] + tau_up[qbest]) / 2
    return tau_prime, __salience(summary_spectrum, fs, tau_low[qbest], 
            tau_up[qbest])

def __hamming_magnitude(fc, fs, N):
    '''
    Calculates the magnitude of a Hamming window's responce to the given 
    frequency.
    @param fc The frequency in question in Hz.
    @param fs The sampling frequency in Hz.
    @param N The size of Hamming window.
    @return The magnitude of response of the Hamming window to the given 
    frequency.
    '''
    # construct the hamming window in the frequency domain
    window = np.fft.fft(__hamming_window(N), n=N)

    # calculate the bin number of the frequency
    bin_width = fs / N
    bin_number = int(fc / bin_width)

    return window[bin_number]

def __extract_notes(summary_spectrum, fs):
    '''
    Extracts the significant F0s from the summary spectrum of a frame.
    @param summary_spectrum A numpy array representing the processed frequency
    spectrum of a signal frame.
    @param fs The sampling frequency in Hz.
    @return A list of detected F0s in Hz.
    '''
    N = summary_spectrum.size
    residual_spectrum = np.copy(summary_spectrum)
    detected = []
    detected_saliences = []

    if __salience(summary_spectrum, fs, 1/FMAX, 1/FMIN) < SILENCE_THRESHOLD:
        return []

    previous_significance = 0
    while True:
        # estimation
        tau, s = __max_salience(residual_spectrum, fs)
        detected_saliences.append(s)
        fc = 1 / tau

        # cancellation
        subtracted_signal = np.zeros(residual_spectrum.size)
        for i in range(1,21):
            partial_frequency = i * fc
            time_vector = np.linspace(0, N / fs, N)
            wave = np.sin(2 * np.pi * partial_frequency * time_vector)

            subtracted_signal = np.add(subtracted_signal,
                    __hamming_magnitude(fc, fs, summary_spectrum.size)
                    * __weight(tau, tau, i, fs) * wave)
        subtracted_signal = abs(np.fft.fft(subtracted_signal, 
                n=subtracted_signal.size))
        fcy_vector = fs * np.arange(0, N) / N
        residual_spectrum = residual_spectrum - subtracted_signal
        residual_spectrum[residual_spectrum<0] = 0

        # check if the F0 is significant
        significance = sum(detected_saliences) / (len(detected_saliences) ** 0.66)
        if significance > previous_significance * SIGINC:
            previous_significance = significance
            detected.append(int(round(librosa.hz_to_midi(fc))))
        else:
            return detected

def __frame_spectrum(signal, fs, onset):
    size_of_frame = int(FRAME_SIZE * fs)
    frame = signal[onset:onset + size_of_frame]
    if frame.size < size_of_frame:
        frame = np.append(frame, np.zeros(size_of_frame - frame.size))
    frame *= __hamming_window(size_of_frame)
    return abs(np.fft.fft(frame, n=size_of_frame*2))

def __add_to_channel(channel, note, time):
    added = False
    for mel in channel:
        if mel.duration > time:
            continue
        if mel.duration < time:
            mel.join(Rest(time - mel.duration))
        mel.join(note)
        added = True
        break
    if not added:
        mel = Melody([Rest(time), note])
        channel.append(mel)

def __extract_pitches_from_onsets(signal, fs):
    filter_fcies = __filter_center_fcies()
    filter_bands = __filter_bandwidths(filter_fcies)
    
    size_of_frame = int(FRAME_SIZE * fs)

    s = np.zeros(signal.size)
    for i in range(0, len(filter_fcies)):
        num_coefs_1, denom_coefs_1, num_coefs_2, denom_coefs_2 = __coefficients(
                filter_fcies[i], fs, filter_bands[i])
        # filter twice with the first filter and twice with the second
        c = sig.lfilter(num_coefs_1, denom_coefs_1, signal)
        c = sig.lfilter(num_coefs_1, denom_coefs_1, c)
        c = sig.lfilter(num_coefs_2, denom_coefs_2, c)
        c = sig.lfilter(num_coefs_2, denom_coefs_2, c)
        # apply neural transduction operations
        c = __dynamic_compression(c, fs)
        c = __fwr_and_filter(c, fs, filter_fcies[i])
        s += c

    # detect onsets
    onsets = librosa.onset.onset_detect(signal, fs, units="samples")

    # detect notes in the frame following every onset
    channels = []
    for i in range(16):
        channels.append([Melody([])])
    for onset in onsets:
        f = __frame_spectrum(s, fs, onset)
        tracked = __extract_notes(f, fs)
        counter = 1
        # track all notes until they are not detected anymore to get durations
        while not len(tracked) == 0:
            f = __frame_spectrum(s, fs, onset + counter * size_of_frame)
            F0s = __extract_notes(f, fs)
            for f0 in tracked:
                if f0 not in F0s:
                    duration = counter * FRAME_SIZE
                    midi_pitch = int(round(librosa.hz_to_midi(f0)))
                    note = Note(duration, midi_pitch, 64)
                    __add_to_channel(channels[0], note, onset / fs)
                    tracked.remove(f0)
    for i in range(len(channels)):
        channels[i] = Channel(Chord(channels[i]), 0, 0)
    return Chord(channels)

def __extract_pitches_per_frame(signal, fs):
    '''
    Polyphonic pitch estimation based on Klapuri 2008
    '''
    filter_fcies = __filter_center_fcies()
    filter_bands = __filter_bandwidths(filter_fcies)

    size_of_frame = int(FRAME_SIZE * fs)
    number_of_frames = int(signal.size / size_of_frame)

    # signal preparation
    s = np.zeros(signal.size)
    for i in range(0, len(filter_fcies)):
        num_coefs_1, denom_coefs_1, num_coefs_2, denom_coefs_2 = __coefficients(
                filter_fcies[i], fs, filter_bands[i])
        # filter twice with the first filter and twice with the second
        c = sig.lfilter(num_coefs_1, denom_coefs_1, signal)
        c = sig.lfilter(num_coefs_1, denom_coefs_1, c)
        c = sig.lfilter(num_coefs_2, denom_coefs_2, c)
        c = sig.lfilter(num_coefs_2, denom_coefs_2, c)
        # apply neural transduction operations
        c = __dynamic_compression(c, fs)
        c = __fwr_and_filter(c, fs, filter_fcies[i])

        # block into frames to apply Hamming windows
        for j in range(0, number_of_frames):
            frame = c[j*size_of_frame:((j+1)*size_of_frame)]
            frame *= __hamming_window(size_of_frame)
            s[j*size_of_frame:((j+1)*size_of_frame)] += frame

    # transform into the frequency domain
    U = np.ndarray([number_of_frames, size_of_frame*2])
    for i in range(0, number_of_frames):
        frame = s[i*size_of_frame:(i+1)*size_of_frame]
        U[i,:] = abs(np.fft.fft(frame, n=frame.size*2))

    # extract F0s from summary spectrum of every frame
    pending = {}
    channels = []
    for i in range(16):
        channels.append([Melody([])])
    for i in range(0, number_of_frames):
        F0s = __extract_notes(U[i,:], fs)
        for f0 in F0s:
            # if the note existed in the previous frames, extend it
            if f0 in pending.keys():
                pending[f0][1].duration += FRAME_SIZE
            # if the note didn't exist before, make a new Note object
            else:
                pending[f0] = (FRAME_SIZE * i, Note(FRAME_SIZE, f0, 64))
        to_be_deleted = []
        for f0 in pending.keys():
            # if an already existing note didn't occur in this frame, the 
            # note has ended
            if f0 not in F0s:
                __add_to_channel(channels[0], pending[f0][1], pending[f0][0])
                to_be_deleted.append(f0)
        for d in to_be_deleted:
            del pending[d]

    for i in range(len(channels)):
        channels[i] = Channel(Chord(channels[i]), i)
    return Chord(channels)

def from_audio(filename, algorithm="frame"):
    signal, fs = librosa.load(filename)
    algorithms = {
            "frame": __extract_pitches_per_frame,
            "onset": __extract_pitches_from_onsets
            }
    return algorithms[algorithm](signal, fs)

def __to_abs_time_per_channel_type_0(midi):
    '''
    Files where all messages are in one track
    '''
    channels = []
    tpb = midi.ticks_per_beat
    for i in range(17):
        channels.append([])
    current_time = 0
    track = midi.tracks[0]
    for msg in track:
        current_time += msg.time / tpb
        msg.time = current_time
        try:
            channels[msg.channel].append(msg)
        except:
            # system-wide messages
            channels[-1].append(msg)
    return channels

def __to_abs_time_per_channel_type_1(midi):
    '''
    Files in which all tracks are played simultaneously
    '''
    channels = []
    tpb = midi.ticks_per_beat
    for i in range(17):
        channels.append([])
    for track in midi.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time / tpb
            msg.time = current_time
            try:
                channels[msg.channel].append(msg)
            except:
                # system-wide messages
                channels[-1].append(msg)
    for i in range(len(channels)):
        channels[i] = sorted(channels[i], key = lambda msg: msg.time)
    return channels

def __parse_channel(events, channel_no, instrument):
    melodies = [Melody([])]
    pending = {}
    def add_to_melodies(start_time, note):
        added = False
        for melody in melodies:
            if melody.duration > start_time:
                continue
            if melody.duration < start_time:
                melody.join(Rest(start_time - melody.duration))
            melody.join(note)
            added = True
            break
        if not added:
            melody = Melody([Rest(start_time), note])
            melodies.append(melody)
    for event in events:
        if event.type == "note_on":
            if event.note in pending.keys():
                start_event = pending[event.note]
                note = Note(0.25, event.note, start_event.velocity)
                add_to_melodies(start_event.time, note)
            pending[event.note] = event
        if event.type == "note_off":
            if event.note not in pending.keys():
                continue
            start_event = pending[event.note]
            note = Note(event.time - start_event.time, event.note, 
                    start_event.velocity)
            add_to_melodies(start_event.time, note)
            del pending[event.note]
    for event in list(pending.values()):
        note = Note(0.25, event.note, event.velocity)
        add_to_melodies(event.time, note)
    return Channel(Chord(melodies), channel = channel_no, instrument = instrument)

def __parse_tempo(channels, instruments, start_time, end_time=-1):
    # filter messages by time bounds
    for i in range(len(channels)):
        if end_time > 0:
            channels[i] = [msg for msg in channels[i] if msg.time >= start_time and msg.time <= end_time]
        else:
            channels[i] = [msg for msg in channels[i] if msg.time >= start_time]

    tempo = 120
    # find the tempo message and set the tempo
    for msg in channels[-1]:
        if msg.type == "set_tempo":
            tempo = mido.tempo2bpm(msg.tempo)
            break
    for i in range(len(channels) - 1):
        channels[i] = __parse_channel(channels[i], channel_no = i,
                instrument = instruments[i])
    return Tempo(Chord(channels[:-1]), tempo)

def __find_channel_instrument(channel):
    for msg in channel:
        if msg.type == "program_change":
            return msg.program
    return 0

def from_midi(filename):
    midi = mido.MidiFile(filename)
    # convert to absolute time in beats and separate by channel
    if midi.type == 0:
        channels = __to_abs_time_per_channel_type_0(midi)
    if midi.type == 1:
        channels = __to_abs_time_per_channel_type_1(midi)
    # for every channel, find instrument
    instruments = list(map(__find_channel_instrument, channels[:-1]))
    # parse every tempo change
    tempo_times = [msg.time for msg in channels[-1] if msg.type == "set_tempo"]
    tempos = []
    # if no tempo changes found, assume constant 120BPM throughout
    if len(tempo_times) == 0:
        tempo_times.append(0)
        channels[-1].insert(0, mido.MetaMessage("set_tempo", time=0, 
            tempo=mido.bpm2tempo(120)))
    for i in range(len(tempo_times) - 1):
        tempos.append(__parse_tempo(channels.copy(), instruments, tempo_times[i], 
            tempo_times[i+1]))
    tempos.append(__parse_tempo(channels.copy(), instruments, tempo_times[-1]))
    # combine all tempo fragmens into one sequence
    return Melody(tempos)
