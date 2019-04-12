import statistics
import typesystem

def all_features(music):
    result = []
    names = []
    for feature in __one_dimensional:
        result.append(feature(music))
        names.append(feature.__name__)
    return names, result

def feature_names():
    return list(map(lambda f: f.__name__, __one_dimensional))

def one_dimensional_features(music):
    return all_features(music)

def multi_dimensional_features(music):
    return [],[]

def overall_dynamic_range(music):
    '''
    The difference between the highest and the lowest velocities.
    '''
    velocities = list(map(lambda note: note.velocity, music.notes()))
    return abs(max(velocities) - min(velocities))

def variation_of_dynamics(music):
    '''
    Standard deviation of velocity.
    '''
    velocities = map(lambda note: note.velocity, music.notes())
    return statistics.stdev(velocities)

def variation_of_dynamics_in_each_voice(music):
    '''
    The mean of standard deviations of velocities for every channel.
    '''
    deviations = []
    def fun(music):
        notes = music.notes()
        if len(notes) > 1:
            deviations.append(statistics.stdev(map(lambda note: note.velocity,
                notes)))
    music.apply(fun, typesystem.Channel, typesystem.Music)
    return statistics.mean(deviations)

def average_note_to_note_dynamics_change(music):
    # TODO
    return 0

__one_dimensional = [
        overall_dynamic_range,
        variation_of_dynamics,
        variation_of_dynamics_in_each_voice,
        average_note_to_note_dynamics_change
        ]
