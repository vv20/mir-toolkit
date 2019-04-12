import operator

PERCUSSION_CHANNEL = 9
PITCH_REVELANCE_THRESHOLD = 0.09
BASS_REGISTER_THRESHOLD = 55
HIGH_REGISTER_THRESHOLD = 73

def __normalise(counts):
    total = sum(counts)
    if total == 0:
        return counts
    return [c / total for c in counts]

def feature_names():
    names = list(map(lambda f: f.__name__, __one_dimensional))
    names.extend(list(map(lambda f: f.__name__, __multi_dimensional)))
    return names

def all_features(music):
    bph = basic_pitch_histogram(music)
    pcd = pitch_class_distribution(music)
    fph = fifths_pitch_histogram(music)
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music, bph=bph, pcd=pcd, fph=fph))
        names.append(feature.__name__)
    for feature in __multi_dimensional:
        feature_value = feature(music, bph=bph, pcd=pcd, fph=fph)
        result.extend(feature_value)
        base_name = feature.__name__
        for i in range(len(feature_value)):
            names.append(base_name + str(i))
    return names, result

def one_dimensional_features(music):
    bph = basic_pitch_histogram(music)
    pcd = pitch_class_distribution(music)
    fph = fifths_pitch_histogram(music)
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music, bph=bph, pcd=pcd, fph=fph))
        names.append(feature.__name__)
    return names, result

def multi_dimensional_features(music):
    bph = basic_pitch_histogram(music)
    pcd = pitch_class_distribution(music)
    fph = fifths_pitch_histogram(music)
    result = []
    names = []
    for feature in __multi_dimensional:
        feature_value = feature(music, bph=bph, pcd=pcd, fph=fph)
        result.extend(feature_value)
        base_name = feature.__name__
        for i in range(len(feature_value)):
            names.append(base_name + str(i))
    return names, result

def most_common_pitch_prevalence(music, bph=None, pcd=None, fph=None):
    '''
    The number of occurrences of the most common pitch.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    return max(bph)

def most_common_pitch_class_prevalence(music, bph=None, pcd=None, fph=None):
    '''
    The number of occurrences of the most common pitch class.
    '''
    if pcd is None:
        pcd = pitch_class_distribution(music)
    return max(pcd)

def relative_strength_of_top_pitches(music, bph=None, pcd=None, fph=None):
    '''
    The ratio of numbers of occurences of the second most common pitch and
    the most common pitch.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    highest = max(bph)
    second_highest = sorted(bph)[-2]
    return second_highest / highest

def relative_strength_of_top_pitch_classes(music, bph=None, pcd=None, fph=None):
    '''
    The ratio of numbers of occurences of the second most common pitch class
    and the most common pitch class.
    '''
    if pcd is None:
        pcd = pitch_class_distribution(music)
    highest = max(pcd)
    second_highest = sorted(pcd)[-2]
    return second_highest / highest

def interval_between_strongest_pitches(music, bph=None, pcd=None, fph=None):
    '''
    Number of semitones between the two most common pitches.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    bph_sorted = []
    for i in range(len(bph)):
        bph_sorted.append((i, bph[i]))
    bph_sorted = sorted(bph_sorted, key = lambda x: x[1])
    return abs(bph_sorted[-1][0] - bph_sorted[-2][0])

def interval_between_strongest_pitch_classes(music, bph=None, pcd=None, fph=None):
    '''
    Number of semitones between the two most common pitch classes.
    '''
    if pcd is None:
        pcd = pitch_class_distribution(music)
    pcd_sorted = []
    for i in range(len(pcd)):
        pcd_sorted.append((i, pcd[i]))
    pcd_sorted = sorted(pcd_sorted, key = lambda x: x[1])
    return abs(pcd_sorted[-1][0] - pcd_sorted[-2][0])

def number_of_common_pitches(music, bph=None, pcd=None, fph=None):
    '''
    Number of pitches that are above the pitch relevance threshold.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    return sum([1 for p in bph if p > PITCH_REVELANCE_THRESHOLD])

def pitch_variety(music, bph=None, pcd=None, fph=None):
    '''
    Number of occurring pitches.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    return sum([1 for p in bph if p > 0])

def pitch_class_variety(music, bph=None, pcd=None, fph=None):
    '''
    Number of occurring pitch classes.
    '''
    if pcd is None:
        pcd = pitch_class_distribution(music)
    return sum([1 for p in pcd if p > 0])

def pitch_range(music, bph=None, pcd=None, fph=None):
    '''
    Number of semitones between the highest and the lowest pitch.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    lowest_index = 0
    highest_index = len(bph) - 1
    while True:
        if bph[lowest_index] == 0:
            lowest_index += 1
        else:
            break
    while True:
        if bph[highest_index] == 0:
            highest_index -= 1
        else:
            break
    return highest_index - lowest_index

def most_common_pitch(music, bph=None, pcd=None, fph=None):
    '''
    Pitch with the highest number of occurrences.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    index, _ = max(enumerate(bph), key=operator.itemgetter(1))
    return index

def primary_register(music, bph=None, pcd=None, fph=None):
    # TODO
    return 0

def importance_of_bass_register(music, bph=None, pcd=None, fph=None):
    '''
    Sum of occurrences of pitches in the bass register.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    return sum(bph[:BASS_REGISTER_THRESHOLD])

def importance_of_middle_register(music, bph=None, pcd=None, fph=None):
    '''
    Sum of occurrences of pitches in the middle register.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    return sum(bph[BASS_REGISTER_THRESHOLD:HIGH_REGISTER_THRESHOLD])

def importance_of_high_register(music, bph=None, pcd=None, fph=None):
    '''
    Sum of occurrences of pitches in the high register.
    '''
    if bph is None:
        bph = basic_pitch_histogram(music)
    return sum(bph[HIGH_REGISTER_THRESHOLD:])

def most_common_pitch_class(music, bph=None, pcd=None, fph=None):
    '''
    Pitch class with the highest number of occurrences.
    '''
    if pcd is None:
        pcd = pitch_class_distribution(music)
    index, _ = max(enumerate(pcd), key=operator.itemgetter(1))
    return index

def dominant_spread(music, bph=None, pcd=None, fph=None):
    # TODO
    return 0

def strong_tonal_centres(music, bph=None, pcd=None, fph=None):
    '''
    Number of local maxima in the fifth pitch histogram. 
    '''
    if fph is None:
        fph = fifths_pitch_histogram(music)
    peaks = 0
    for i in range(len(fph)):
        left = i - 1
        right = (i + 1) % len(fph)
        if fph[i] > fph[left] and fph[i] > fph[right]:
            peaks += 1
    return peaks

def basic_pitch_histogram(music, bph=None, pcd=None, fph=None):
    '''
    Occurrences of pitches, normalised to sum to one.
    '''
    if bph is not None:
        return bph
    notes = music.notes()
    counts = [0] * 128
    for note in notes:
        counts[note.pitch] += 1
    return __normalise(counts)

def pitch_class_distribution(music, bph=None, pcd=None, fph=None):
    if pcd is not None:
        return pcd
    notes = music.notes()
    counts = [0] * 12
    for note in notes:
        counts[note.pitch % 12] += 1
    return __normalise(counts)

def fifths_pitch_histogram(music, bph=None, pcd=None, fph=None):
    if pcd is None:
        pcd = pitch_class_distribution(music)
    counts = [0] * 12
    for i in range(0, 12):
        counts[7*i%12] = pcd[i]
    return counts

def quality(music, bph=None, pcd=None, fph=None):
    # TODO
    return 0

def glissandro_prevalence(music, bph=None, pcd=None, fph=None):
    # TODO
    return 0

def average_range_of_glissandros(music, bph=None, pcd=None, fph=None):
    # TODO
    return 0

def vibrato_prevalence(music, bph=None, pcd=None, fph=None):
    # TODO
    return 0

__one_dimensional = [
        most_common_pitch_prevalence,
        most_common_pitch_class_prevalence,
        relative_strength_of_top_pitches,
        relative_strength_of_top_pitch_classes,
        interval_between_strongest_pitches,
        interval_between_strongest_pitch_classes,
        number_of_common_pitches,
        pitch_variety,
        pitch_class_variety,
        pitch_range,
        most_common_pitch,
        primary_register,
        importance_of_bass_register,
        importance_of_middle_register,
        importance_of_high_register,
        most_common_pitch_class,
        dominant_spread,
        strong_tonal_centres,
        quality,
        glissandro_prevalence,
        average_range_of_glissandros,
        vibrato_prevalence
        ]

__multi_dimensional = [
        basic_pitch_histogram,
        pitch_class_distribution,
        fifths_pitch_histogram
        ]
