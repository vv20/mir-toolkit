import operator

from typesystem import Melody, Primitive, Note

INTERVAL_RELEVANCE_THRESHOLD = 0.09

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
    mih = melodic_interval_histogram(music)
    amih = absolute_melodic_interval_histogram(music)
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music, mih=mih, amih=amih))
        names.append(feature.__name__)
    for feature in __multi_dimensional:
        feature_value = feature(music, mih=mih)
        result.extend(feature_value)
        base_name = feature.__name__
        for i in range(len(feature_value)):
            names.append(base_name + str(i))
    return names, result

def one_dimensional_features(music):
    mih = melodic_interval_histogram(music)
    amih = absolute_melodic_interval_histogram(music)
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music, mih=mih, amih=amih))
        names.append(feature.__name__)
    return names, result

def multi_dimensional_features(music):
    mih = melodic_interval_histogram(music)
    amih = absolute_melodic_interval_histogram(music)
    result = []
    names = []
    for feature in __multi_dimensional:
        feature_value = feature(music, mih=mih)
        result.extend(feature_value)
        base_name = feature.__name__
        for i in range(len(feature_value)):
            names.append(base_name + str(i))
    return names, result

def melodic_interval_histogram(music, mih=None, amih=None):
    if mih is not None:
        return mih
    counts = [0] * 256
    def fun(music):
        prev_note = None
        for music in music.musics:
            if not isinstance(music, Note):
                continue
            if prev_note is None:
                prev_note = music
                continue
            difference = music.pitch - prev_note.pitch
            counts[difference] += 1
            prev_note = music
    music.apply(fun, Melody, Primitive)
    return __normalise(counts)

def absolute_melodic_interval_histogram(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    counts = [0] * 128
    counts[0] = mih[0]
    for i in range(1, len(counts)):
        counts[i] = mih[i] + mih[-i]
    return __normalise(counts)

def average_melodic_interval(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    total = 0
    for i in range(len(amih)):
        total += i * amih[i]
    if total == 0:
        return total
    return total / sum(amih)

def most_common_melodic_interval(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    index, _ = max(enumerate(mih), key = operator.itemgetter(1))
    return index

def distance_between_most_common_melodic_intervals(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    amih_sorted = []
    for i in range(len(amih)):
        amih_sorted.append((i, amih[i]))
    amih_sorted = sorted(amih_sorted, key = lambda x: x[1])
    return abs(amih_sorted[-1][0] - amih_sorted[-2][0])

def most_common_melodic_interval_prevalence(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    return max(mih)

def relative_strength_of_most_common_intervals(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    highest = max(mih)
    second_highest = sorted(mih)[-2]
    if highest == 0:
        return 0
    return second_highest / highest

def number_of_common_melodic_intervals(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    return sum([1 for i in mih if i > INTERVAL_RELEVANCE_THRESHOLD])

def amount_of_arpeggiation(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    result = mih[0]
    result += mih[3]
    result += mih[4]
    result += mih[7]
    result += mih[10]
    result += mih[11]
    result += mih[12]
    result += mih[15]
    result += mih[16]
    return result

def repeated_notes(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    return mih[0]

def chromatic_motion(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    return amih[1]

def stepwise_motion(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    return amih[1] + amih[2]

def melodic_thirds(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    return amih[3] + amih[4]

def melodic_fifths(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    return amih[7]

def melodic_tritones(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    return amih[6]

def melodic_octaves(music, mih=None, amih=None):
    if amih is None:
        amih = absolute_melodic_interval_histogram(music)
    return amih[12]

def direction_of_motion(music, mih=None, amih=None):
    if mih is None:
        mih = melodic_interval_histogram(music)
    ups = sum(mih[1:128])
    downs = sum(mih[-128:-1])
    return ups - downs

def duration_of_melodic_arcs(music, mih=None, amih=None):
    # TODO
    return 0

def size_of_melodic_arcs(music, mih=None, amih=None):
    # TODO
    return 0

__one_dimensional = [
        average_melodic_interval,
        most_common_melodic_interval,
        distance_between_most_common_melodic_intervals,
        most_common_melodic_interval_prevalence,
        relative_strength_of_most_common_intervals,
        number_of_common_melodic_intervals,
        amount_of_arpeggiation,
        repeated_notes,
        chromatic_motion,
        stepwise_motion,
        melodic_thirds,
        melodic_fifths,
        melodic_tritones,
        melodic_octaves,
        direction_of_motion,
        duration_of_melodic_arcs,
        size_of_melodic_arcs,
        ]

__multi_dimensional = [
        melodic_interval_histogram,
        absolute_melodic_interval_histogram,
        ]
