# MIR-Toolkit
A library for music manipulation and analysis. The underlying algorithms and
reasoning behind the library are descibed in [5].

## Music objects
All objects in the data model of MIR-Toolkit are a subclass of the Music abstract
class and may be found in the typesystem module. All objects have a duration.

### Primitives
There are two primitive objects in the type system, that both inherit from the
Primitive abstract class, that in turn inherits from the Music abstract class.
These are Note and Rest.

A Note has a duration (in beats), a (MIDI) pitch, and a (MIDI) velocity:
```python
from typesystem import Note
n = Note(1, 69, 64)
```

A Rest only has a duration (in beats):
```python
from typesystem import Rest
r = Rest(4)
```

### Combinations
There are two types of object combinations, that both inherit from the Combination
abstract class, that in turn inherits from the Music class. They are Melody and
Chord. Both take as an argument to the constructor a list of Music objects that
they are to be applied to. The duration of combinations is calculated as the
sum of durations of underlying objects for Melody, and as the maximum of durations
of underlying objects for Chord.

```python
from typesystem import Note, Chord, Melody
a = Note(1, 69, 64)
c = Note(1, 73, 64)
e = Note(1, 76, 64)
a2 = Note(2, 81, 64)
arpeggio = Melody([a,c,e,a2])
chord = Chord([a,c,e,a2])
print(arpeggio.duration) # 5
print(chord.duration) #2
```

NB. When creating a Melody, order of the underlying Music objects is important,
as they shall be played sequentially. When creating a Chord, the order does not
matter, since they are all played simultaneously.

### Encapsulations
Encapsulations are Music objects that have one child (underlying Music object) and
add properties to it. The two Encapsulation classes currently implemented are
Channel, that allows to add a MIDI channel and a MIDI instrument, and Tempo,
that assigns a tempo (in beats per minute) to the underlying object and converts
time from beats per minute to seconds. They both inherit from the Encapsulation
class, that inherits from the Music class.

```python
from typesystem import Note, Melody, Channel, Tempo
n1 = Note(2, 69, 64)
n2 = Note(2, 81, 64)
m = Melody([n1, n2])
channel = Channel(m, 5, 74) # assign the created melody to MIDI channel 5, to
                            # be played on a flute (instrument number 74)
tempo = Tempo(channel, 150) # play the created piece at 150 BPM
print(tempo.duration)       # note that this is still 5 since duration is counted
                            # beats and is unaffected by tempo
```

## Extraction of Music Objects
One may wish to obtain a symbolic representation of music in the given type system
from a MIDI file or a music recording. Functionality for that is included in the
music\_input module.

```python
import music_input
midi = music_input.from_midi("filename.mid")
audio = music_input.from_audio("filename.wav")
```

For extraction of music from recordings, the transcription algorithm specified
in [1] was used. There are two versions of the transcription algorithm available- 
one analyses the recording frame-by-frame, the other detects note onsets first and 
then extracts pitches starting at onset frames. The algorithm can be specified 
using the argument "algorithm":
```python
import music_input
m1 = music_input.from_audio("filename.wav", algorithm="frame")
m2 = music_input.from_audio("filename.wav", algorithm="onset")
```

Transcription algorithms were tested using the Classical Piano MIDI dataset [4].

## Onset Detection
As a tool for transcription and other purposes, MIR-Toolkit provides a suite of
onset extraction algorithms, as specified in [2]. These can be used as follows:

```python
import librosa
import onset_detection
signal, fs = librosa.load("filename.wav")
onsets = onset_detection.detect_onsets(signal, fs)
```

Optional arguments to "detect\_onsets" are:
- rt (reduction function)- may be one of "hfc"- high-frequency content, "sd"- spectral difference or "pd"- phase difference. The default is "hfc".
- hop\_length- an integer that specifies number of samples per frame of analysis. The default is 512.
- units- may be one of "samples", "frames" or "time". The detected onset times are returned measured in the specified unit of time (in case of "time" it is seconds). The default is "samples".
- split\_bands- a boolean specifying whether the signal should be split into multiple frequency bands pre-analysis or not. The default is False.
- threshold- an integer specifying the threshold value that the reduction function must reach for an onset candidate to be considered. The default is 0.

Onset detection algorithms were tested using the Onset Database dataset [3].

## Function Application
All Music objects have a function "apply" that takes the arguments "fun", "class1"
and "class2" and applies the function "fun" to all elements of the object tree if
they are instances of "class1" and their children are instances of "class2". This
allows a user to, for example, counts the numbers of notes in every channel:

```python
import music_input
from typesystem import Music, Channel
m = music_input.from_midi("filename.mid")
counts = [0] * 16
def fun(music):
  counts[music.channel] += len(music.notes())
m.apply(fun, Channel, Music)
```

In addition, all Music objects have a "notes" convenience function that recursively
extracts all notes from the underlying objects, and a "pitch\_intervals" function
that extracts a list of dictionaries that contain the absolute onset time, absolute
offset time and the pitch of every note. The latter is used for testing of
transcription algorithms.

## Feature Extraction
Having acquired a Music object that represents a composition, one may wish to
calculate some feature values based on it. The feature extraction functions are
contained in the "feature\_extraction" module, that has four submodules:
- dynamic\_features
- instrumentation\_features
- melodic\_features
- pitch\_features

A user may extract features values one by one:

```python
import feature_extraction.pitch_features
import music_input
m = music_input.from_midi("filename.mid")
print(feature_extraction.pitch_features.pitch_class_distribution(m))
```

Or obtain a list of all feature names and values from a module by calling one of
the three aggregate feature functions:

```python
import feature_extraction.pitch_features
import music_input
m = music_input.from_midi("filename.mid")
names, values = feature_extraction.pitch_features.all_features(m)
oneDnames, oneDvalues = feature_extraction.pitch_features.one_dimensional_features(m)
multiDnames, multiDvalues = feature_extraction.pitch_features.multi_dimensional_features(m)
```

These functions calculate the names and values of all features, one one dimensional
features, and only multi-dimensional features respectively.

## References:
1. Klapuri, A (2008). "Multipitch Analysis of Polyphonic Music and Speech Signals Using an Auditory Model".
2. Bello, J. P. et al. (2005). "A tutorial on onset detection in music signals".
3. Onset Database [https://grfia.dlsi.ua.es/cm/worklines/pertusa/onset/ODB/](https://grfia.dlsi.ua.es/cm/worklines/pertusa/onset/ODB/)
4. Classical Piano MIDI [http://www.piano-midi.de/](http://www.piano-midi.de/)
5. Vasilyev, V (2019). "MIR-Toolkit: Framework for Music Information Retrieval Research"
