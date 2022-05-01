from typing import List
from pprint import pprint

from mido import MidiFile, MidiTrack, second2tick, Message, MetaMessage


class Chord:
    types = {
        'major': {
            'triad': (0, 4, 7),
            'first_inversions': (4, 7, 12),
            'second_inversions': (7, 12, 16)
        },
        'minor': {
            'triad': (0, 3, 7),
            'first_inversions': (3, 7, 12),
            'second_inversions': (7, 12, 15)
        },
        'diminished': (0, 3, 6),
        'sus2': (0, 2, 7),
        'sus4': (0, 5, 7)
    }

    def __init__(self,
                 name: str,
                 root: int,
                 chord_type: str):
        self.name = name
        self.key = root
        self.chord_type = chord_type

        if chord_type == 'minor':
            self.name += 'm'

    def get_triad(self) -> List[int]:
        return [self.key + i for i in Chord.types[self.chord_type]['triad']]

    def get_first_inversions(self) -> List[int]:
        return [self.key + i for i in Chord.types[self.chord_type]['first_inversions']]

    def get_second_inversions(self) -> List[int]:
        return [self.key + i for i in Chord.types[self.chord_type]['second_inversions']]

    def get_diminished(self) -> List[int]:
        return [self.key + i for i in Chord.types['diminished']]

    def get_sus2(self) -> List[int]:
        return [self.key + i for i in Chord.types['sus2']]

    def get_sus4(self) -> List[int]:
        return [self.key + i for i in Chord.types['sus4']]

    def __repr__(self):
        return self.name


class Keys:
    names = ["C", "C|D", "D", "D|E", "E", "F", "F|G", "G", "G|A", "A", "A|H", "H"]
    major_sequence = (0, 2, 4, 5, 7, 9, 11)
    minor_sequence = (0, 2, 3, 5, 7, 8, 10)

    def __init__(self, melody: MidiFile):
        self.melody = melody

        self.keys = {}
        for ind, key_name in enumerate(Keys.names):
            self.keys[key_name] = []
            for step, delta in enumerate(Keys.major_sequence):
                val = (ind + delta) % 12
                if step + 1 in [2, 3, 6]:
                    self.keys[key_name].append(Chord(Keys.names[val], val, 'minor'))
                else:
                    self.keys[key_name].append(Chord(Keys.names[val], val, 'major'))

        for ind, key_name in enumerate(Keys.names):
            self.keys[key_name + "m"] = []
            for step, delta in enumerate(Keys.minor_sequence):
                val = (ind + delta) % 12
                if step + 1 in [1, 4, 5]:
                    self.keys[key_name + "m"].append(Chord(Keys.names[val], val, 'minor'))
                else:
                    self.keys[key_name + "m"].append(Chord(Keys.names[val], val, 'major'))

    def get_melody_key(self) -> str:
        self.notes = [message.note % 12 for message in self.melody if message.type == 'note_on']
        self.notes_without_repeating = list(set(self.notes))

        keys_matching_notes = {}
        for key_name, chords in self.keys.items():
            count = 0
            for chord in chords:
                if chord.key in self.notes_without_repeating:
                    count += 1
            if count in keys_matching_notes:
                keys_matching_notes[count].append(key_name)
            else:
                keys_matching_notes[count] = [key_name]

        max_matches = max(keys_matching_notes)
        keys_priority = {}
        for key_name in keys_matching_notes[max_matches]:
            keys_priority[key_name] = 0

        stable_notes = {}
        for key_name in keys_matching_notes[max_matches]:
            stable_notes[key_name] = [
                self.keys[key_name][0].name.replace('m', ''),
                self.keys[key_name][2].name.replace('m', ''),
                self.keys[key_name][3].name.replace('m', ''),
                self.keys[key_name][4].name.replace('m', ''),
                0
            ]

        for key_name in keys_priority:
            if Keys.names[self.notes[-1]] == stable_notes[key_name][0]:
                keys_priority[key_name] += 1
            if Keys.names[self.notes[-1]] == stable_notes[key_name][2] \
                    or Keys.names[self.notes[-1]] == stable_notes[key_name][3]:
                keys_priority[key_name] += 0.66
            if Keys.names[self.notes[-1]] == stable_notes[key_name][1]:
                keys_priority[key_name] += 0.33

            if Keys.names[self.notes[0]] == stable_notes[key_name][0]:
                keys_priority[key_name] += 1
            if Keys.names[self.notes[0]] == stable_notes[key_name][2] \
                    or Keys.names[self.notes[0]] == stable_notes[key_name][3]:
                keys_priority[key_name] += 0.66
            if Keys.names[self.notes[0]] == stable_notes[key_name][1]:
                keys_priority[key_name] += 0.33

        for note in self.notes:
            for key_name in keys_priority:
                if Keys.names[note] in stable_notes[key_name]:
                    stable_notes[key_name][-1] += 1

        max_stable_notes, key_name_with_max_stable_matches = 0, ''
        for key_name in keys_priority:
            if max_stable_notes < stable_notes[key_name][-1]:
                key_name_with_max_stable_matches = key_name
                max_stable_notes = stable_notes[key_name][-1]

        keys_priority[key_name_with_max_stable_matches] += 1

        max_priority = max(keys_priority.values())

        return list(keys_priority.keys())[list(keys_priority.values()).index(max_priority)]


class EvolutionAlgorithm:
    pass

class AccompanimentGenerator:
    def __init__(self, path: str):
        self.initial_melody = MidiFile(path, clip=True)

    def generate(self) -> MidiFile:
        self.with_accompaniment = self.initial_melody
        keys = Keys(self.initial_melody)
        melody_key = keys.get_melody_key()
        print(f"Melody key is {melody_key}")

        # TODO create the evolution and generate the chords

        return self.with_accompaniment


melody_path = "barbiegirl_mono.mid"
melody_with_accompaniment_path = "barbiegirl_mono_accompaniment"
accompaniment = AccompanimentGenerator(melody_path).generate()
accompaniment.save(melody_with_accompaniment_path)
