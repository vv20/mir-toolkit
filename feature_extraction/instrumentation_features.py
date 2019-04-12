import statistics

import typesystem

PERCUSSION_CHANNEL = 9
STRING_KEYBOARD_GROUP = list(range(8))
ACOUSTIC_GUITAR_GROUP = [24,25]
ELECTRIC_GUITAR_GROUP = list(range(26,32))
VIOLIN_GROUP = [40,110]
SAXOPHONE_GROUP = list(range(64,68))
BRASS_GROUP = list(range(56,68))
WOODWINDS_GROUP = list(range(68,76))
ORCHESTRAL_STRINGS_GROUP = list(range(40,47))
STRING_ENSEMBLE_GROUP = list(range(48,52))
ELECTRIC_INSTRUMENT_GROUP = [4,5,16,18,26,27,28,29,30,31,33,34,35,36,37,38,39]

def __instrument_group_note_count(instruments, music):
    notes = []
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL and music.instrument in instruments:
            notes.extend(music.notes())
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return len(notes)

def feature_names():
    names = list(map(lambda f: f.__name__, __one_dimensional))
    names.extend(list(map(lambda f: f.__name__, __multi_dimensional)))
    return names

def all_features(music):
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music))
        names.append(feature.__name__)
    for feature in __multi_dimensional:
        feature_value = feature(music)
        result.extend(feature_value)
        base_name = feature.__name__
        for i in range(len(feature_value)):
            names.append(base_name + str(i))
    return names, result

def one_dimensional_features(music):
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music))
        names.append(feature.__name__)
    return names, result

def multi_dimensional_features(music):
    result = []
    names = []
    for feature in __multi_dimensional:
        feature_value = feature(music)
        result.extend(feature_value)
        base_name = feature.__name__
        for i in range(len(feature_value)):
            names.append(base_name + str(i))
    return names, result

def pitched_instruments_present(music):
    present = [0] * 128
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL:
            present[music.instrument] = 1
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return present

def unpitched_instruments_present(music):
    present = [0] * 128
    def fun(music):
        if music.channel == PERCUSSION_CHANNEL:
            for note in music.notes():
                present[note.pitch] = 1
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return present

def note_prevalence_of_pitched_instruments(music):
    counts = [0] * 128
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL:
            counts[music.instrument] += len(music.notes())
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return counts

def note_prevalence_of_unpitched_instruments(music):
    counts = [0] * 128
    def fun(music):
        if music.channel == PERCUSSION_CHANNEL:
            for note in music.notes():
                counts[note.pitch] += 1
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return counts

def time_prevalence_of_pitched_instruments(music):
    counts = [0] * 128
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL:
            total_duration = sum(map(lambda n: n.duration, music.notes()))
            counts[music.instrument] += total_duration
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return counts

def variability_of_note_prevalence_of_pitched_instruments(music):
    instrument_counts = [0] * 128
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL:
            instrument_counts[music.instrument] += len(music.notes())
    music.apply(fun,typesystem.Channel, typesystem.Music)

    instrument_counts = [ic for ic in instrument_counts if ic > 0]
    if len(instrument_counts) < 2:
        return 0
    return statistics.stdev(instrument_counts)

def variability_of_note_prevalence_of_unpitched_instruments(music):
    instrument_counts = [0] * 128
    def fun(music):
        if music.channel == PERCUSSION_CHANNEL:
            for note in music.notes():
                instrument_counts[note.pitch] += 1
    music.apply(fun, typesystem.Channel, typesystem.Music)

    instrument_counts = [ic for ic in instrument_counts if ic > 0]
    if len(instrument_counts) < 2:
        return 0
    return statistics.stdev(instrument_counts)

def number_of_pitched_instruments(music):
    instruments = []
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL:
            instruments.append(music.instrument)
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return len(set(instruments))

def number_of_unpitched_instruments(music):
    instruments = []
    def fun(music):
        if music.channel == PERCUSSION_CHANNEL:
            instruments.extend(map(lambda n: n.pitch, music.notes()))
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return len(set(instruments))

def percussion_prevalence(music):
    perc_notes = []
    non_perc_notes = []
    def fun(music):
        if music.channel != PERCUSSION_CHANNEL:
            non_perc_notes.extend(music.notes())
    music.apply(fun, typesystem.Channel, typesystem.Music)
    def fun(music):
        if music.channel == PERCUSSION_CHANNEL:
            perc_notes.extend(music.notes())
    music.apply(fun, typesystem.Channel, typesystem.Music)
    perc_number = len(perc_notes)
    non_perc_number = len(non_perc_notes)
    return perc_number / (perc_number + non_perc_number)

def string_keyboard_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(STRING_KEYBOARD_GROUP, music) / total

def acoustic_guitar_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(ACOUSTIC_GUITAR_GROUP, music) / total

def electric_guitar_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(ELECTRIC_GUITAR_GROUP, music) / total

def violin_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(VIOLIN_GROUP, music) / total

def saxophone_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(SAXOPHONE_GROUP, music) / total

def brass_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(BRASS_GROUP, music) / total

def woodwinds_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(WOODWINDS_GROUP, music) / total

def orchestral_strings_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(ORCHESTRAL_STRINGS_GROUP, music) / total

def string_ensemble_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(STRING_ENSEMBLE_GROUP, music) / total

def electric_instrument_fraction(music):
    total = len(music.notes())
    return __instrument_group_note_count(ELECTRIC_INSTRUMENT_GROUP, music) / total

__one_dimensional = [
        variability_of_note_prevalence_of_pitched_instruments,
        variability_of_note_prevalence_of_unpitched_instruments,
        number_of_pitched_instruments,
        number_of_unpitched_instruments,
        percussion_prevalence,
        string_keyboard_fraction,
        acoustic_guitar_fraction,
        electric_guitar_fraction,
        violin_fraction,
        saxophone_fraction,
        brass_fraction,
        woodwinds_fraction,
        orchestral_strings_fraction,
        string_ensemble_fraction,
        electric_instrument_fraction
        ]

__multi_dimensional = [
        pitched_instruments_present,
        unpitched_instruments_present,
        note_prevalence_of_pitched_instruments,
        note_prevalence_of_unpitched_instruments,
        time_prevalence_of_pitched_instruments
        ]
