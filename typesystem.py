from abc import ABC, abstractmethod
from enum import Enum
from mido import Message
import mido
from operator import itemgetter

class Music(ABC):
    def __init__(self, duration):
        self.duration = duration

    @abstractmethod
    def to_midi(self):
        '''
        Convert the music to a performance, i.e. a list of MIDI events
        that corresponds to the underlying music piece.
        @return a list of mido Message objects.
        '''
        pass

    def join(self, music):
        '''
        Append another music piece to this piece to create a sequence
        of music objects (melody).
        @param music The music to be appended to this one.
        '''
        return Melody(self, music)

    def layer(self, music):
        '''
        Layer another music piece with this one to create a simultaneity
        of music objects (chord).
        @param music The music to be layered with this one.
        '''
        return Chord(self, music)

    @abstractmethod
    def notes(self, start_time):
        pass

    @abstractmethod
    def pitch_intervals(self, start_time=0):
        pass

    def apply(self, fun, self_class, children_class):
        pass

class Primitive(Music):
    def __init__(self, duration):
        super().__init__(duration)

    def apply(self, fun, self_class, children_class):
        if isinstance(self, self_class):
            fun(self)

class Combination(Music):
    def __init__(self, duration, musics):
        super().__init__(duration)
        self.musics = musics
        for i in range(len(musics)):
            if not isinstance(musics[i], Music):
                raise TypeError

    def notes(self):
        notes = []
        for m in self.musics:
            notes.extend(m.notes())
        return notes

    def apply(self, fun, self_class, children_class):
        applicable = True
        if not isinstance(self, self_class):
            applicable = False
        for music in self.musics:
            music.apply(fun, self_class, children_class)
            if not isinstance(music, children_class):
                applicable = False
                break
        if applicable:
            fun(self)

class Encapsulation(Music):
    def __init__(self, music):
        if not isinstance(music, Music):
            raise TypeError
        super().__init__(music.duration)
        self.music = music

    def notes(self):
        return self.music.notes()

    def apply(self, fun, self_class, children_class):
        if isinstance(self, self_class) and isinstance(self.music, children_class):
            fun(self)
        self.music.apply(fun, self_class, children_class)

class Rest(Primitive):
    def __init__(self, duration):
        '''
        Duration in beats
        '''
        super().__init__(duration)

    def to_midi(self):
        start = mido.Message("note_on", time = 0, note = 0)
        end = mido.Message("note_off", time = self.duration,
                note = 0, velocity = 0)
        return [start, end]

    def notes(self):
        return []

    def pitch_intervals(self, start_time=0):
        return []

class Note(Primitive):
    def __init__(self, duration, pitch, velocity):
        '''
        Duration in beats
        '''
        super().__init__(duration)
        self.pitch = pitch
        self.velocity = velocity

    def to_midi(self):
        start = mido.Message("note_on", time = 0, note = self.pitch,
                velocity = self.velocity)
        end = mido.Message("note_off", time = self.duration,
                note = self.pitch, velocity = 0)
        return [start, end]

    def notes(self):
        return [self]

    def pitch_intervals(self, start_time=0):
        return [{"onset": start_time, "offset": start_time + self.duration,
            "pitch": self.pitch}]

class Melody(Combination):
    def __init__(self, musics):
        if len(musics) == 0:
            super().__init__(0, musics)
        else:
            super().__init__(sum([m.duration for m in musics]), musics)

    def join(self, music):
        self.duration += music.duration
        self.musics.append(music)

    def to_midi(self):
        events = []
        for m in self.musics:
            events.extend(m.to_midi())
        new_events = []
        for i in range(len(events)):
            try:
                if events[i].note == 0 and i < len(events) - 1:
                    events[i+1].time += events[i].time
                else:
                    new_events.append(events[i])
            except AttributeError:
                new_events.append(events[i])
        return new_events

    def pitch_intervals(self, start_time=0):
        notes = []
        for m in self.musics:
            notes.extend(m.pitch_intervals(start_time))
            start_time += m.duration
        return notes

class Chord(Combination):
    def __init__(self, musics):
        if len(musics) == 0:
            super().__init__(0, musics)
        else:
            super().__init__(max([m.duration for m in musics]), musics)

    def layer(self, music):
        self.duration = max(self.duration, music.duration)
        self.musics.append(music)

    def to_midi(self):
        events_by_component = []
        for m in self.musics:
            events_by_component.append(m.to_midi())
        # convert to absolute time
        for ebc in events_by_component:
            time = 0
            for e in ebc:
                e.time = time + e.time
                time = e.time
        events = []
        for ebc in events_by_component:
            events.extend(ebc)
        events = sorted(events, key=lambda e: e.time)
        # and back to delta time
        time = 0
        for e in events:
            temp_time = e.time
            e.time = e.time - time
            time = temp_time
        return events

    def notes(self):
        notes = []
        for m in self.musics:
            notes.extend(m.notes())
        return notes

    def pitch_intervals(self, start_time=0):
        notes = []
        for m in self.musics:
            notes.extend(m.pitch_intervals(start_time))
        return notes

class Channel(Encapsulation):
    def __init__(self, music, channel, instrument):
        super().__init__(music)
        self.channel = channel
        self.instrument = instrument

    def to_midi(self):
        events = self.music.to_midi()
        for e in events:
            try:
                e.channel = self.channel
            except AttributeError:
                pass
        events.insert(0,
                mido.Message("program_change", time=0, program=self.instrument))
        return events

    def pitch_intervals(self, start_time=0):
        return self.music.pitch_intervals(start_time)

class Tempo(Encapsulation):
    def __init__(self, music, tempo):
        '''
        Takes tempo in BPM
        '''
        super().__init__(music)
        self.tempo = tempo

    def to_midi(self):
        events = [mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(self.tempo))]
        events.extend(self.music.to_midi())
        return events

    def pitch_intervals(self, start_time=0):
        intervals = self.music.pitch_intervals(start_time)
        new_intervals = []
        for interval in intervals:
            interval["onset"] = interval["onset"] / self.tempo * 60
            interval["offset"] = interval["offset"] / self.tempo * 60
            new_intervals.append(interval)
        return new_intervals
