import random
from typing import List, Tuple, Union
import numpy as np

from mido import MidiFile, MidiTrack, Message, MetaMessage


class Chord:
    """
    Chord class for creating custom-named chords and selecting any existing type
    """
    names = ['triad', 'diminished', 'sus2', 'sus4']  # names of types

    # Types with sequence of note intervals
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
                 chord_type: str,
                 is_only_dim: bool = False
                 ):
        """
        Chord constructor for initializing the data
        :param name: name of chord
        :param root: the number of the note that is the root of the chord
        :param chord_type: major or minor chord
        :param is_only_dim: if the chord is diminished
        """
        self.name = name
        self.key = root
        self.chord_type = chord_type
        self.is_only_dim = is_only_dim

        if chord_type == 'minor':
            self.name += 'm'

    def get_notes(self) -> List[int]:
        """
        Get notes of thriad chord (not depends on inversions)
        :return: list contains value of notes
        """
        return [(self.key + i) % 12 for i in Chord.types[self.chord_type]['triad']]

    def get_triad(self) -> List[int]:
        """
        Get thriad chord
        :return: list contains value of notes
        """
        return [self.key + i for i in Chord.types[self.chord_type]['triad']]

    def get_first_inversions(self) -> List[int]:
        """
        Get first inversion (Sextachord)
        :return: list contains value of notes
        """
        return [self.key + i for i in Chord.types[self.chord_type]['first_inversions']]

    def get_second_inversions(self) -> List[int]:
        """
        Get second inversion (Quartsuxtachcord)
        :return: list contains value of notes
        """
        return [self.key + i for i in Chord.types[self.chord_type]['second_inversions']]

    def get_diminished(self) -> List[int]:
        """
        Get diminished chord
        :return: list contains value of notes
        """
        return [(self.key + i) % 12 for i in Chord.types['diminished']]

    def get_sus2(self) -> List[int]:
        """
        Get suspended second chord
        :return: list contains value of notes
        """
        return [(self.key + i) % 12 for i in Chord.types['sus2']]

    def get_sus4(self) -> List[int]:
        """
        Get suspended fourth chord
        :return: list contains value of notes
        """
        return [(self.key + i) % 12 for i in Chord.types['sus4']]

    def __repr__(self):
        """
        Represent the name of chord for debugging
        :return: name of chord
        """
        return self.name


class Keys:
    """
    Keys determination class
    """
    names = ["C", "C|D", "D", "D|E", "E", "F", "F|G", "G", "G|A", "A", "A|H", "H"]
    major_sequence = (0, 2, 4, 5, 7, 9, 11)
    minor_sequence = (0, 2, 3, 5, 7, 8, 10)

    def __init__(self, melody: MidiFile):
        """
        Keys class constructor, which generates a set of all possible major and minor keys
        with chords that applicable in each key
        :param melody: initial melody
        """
        self.melody = melody
        self.notes = [message.note % 12 for message in self.melody if message.type == 'note_on']
        self.notes_without_repeating = list(set(self.notes))

        self.keys = {}
        for ind, key_name in enumerate(Keys.names):
            self.keys[key_name] = []
            for step, delta in enumerate(Keys.major_sequence):
                val = (ind + delta) % 12
                if step + 1 in [2, 3, 6]:
                    self.keys[key_name].append(Chord(Keys.names[val], val, 'minor'))
                else:
                    if step + 1 == 7:
                        self.keys[key_name].append(Chord(Keys.names[val], val, 'major', True))
                    else:
                        self.keys[key_name].append(Chord(Keys.names[val], val, 'major'))

        for ind, key_name in enumerate(Keys.names):
            self.keys[key_name + "m"] = []
            for step, delta in enumerate(Keys.minor_sequence):
                val = (ind + delta) % 12
                if step + 1 in [1, 4, 5]:
                    self.keys[key_name + "m"].append(Chord(Keys.names[val], val, 'minor'))
                else:
                    if step + 1 == 2:
                        self.keys[key_name + "m"].append(Chord(Keys.names[val], val, 'major', True))
                    else:
                        self.keys[key_name + "m"].append(Chord(Keys.names[val], val, 'major'))

    def get_melody_key(self) -> Tuple[str, list]:
        """
        Get the key of initial melody
        :return: the key name and a list of the names of all chords in the computed key
        """
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

        melody_key = list(keys_priority.keys())[list(keys_priority.values()).index(max_priority)]
        return melody_key, self.keys[melody_key]


def contains_sequence(sequence: List[Union[str, int]],
                      in_sequence: List[Union[str, int]]
                      ):
    """
    Check is first sequence of note values contains in the second sequence of note values
    :param sequence: sequence of note values
    :param in_sequence: sequence of note values
    :return: True if it is containing, False otherwise
    """
    in_sequence_str = ''.join(list(map(str, in_sequence)))
    sequence_str = ''.join(list(map(str, sequence)))
    return sequence_str in in_sequence_str


class EvolutionAlgorithm:
    """
    Evolution Algorithm class working by genetic algorithm
    """
    COUNT_WITHOUT_CHANGING = 20

    def __init__(self,
                 generations: int,
                 population_size: int,
                 key: str,
                 chords: List[Chord]
                 ):
        """
        Evolution Algorithm class constructor
        :param generations: number of total generations
        :param population_size: number of individuals in population
        :param key: key of the melody
        :param chords: applicable chords for such key
        """
        self.generations = generations
        self.population_size = population_size
        self.key = key
        self.chords = chords

    def get_individual(self) -> Tuple[Chord, List[int]]:
        """
        Create the individual (random chord)
        :return: tuple of chord and notes values of chord
        """
        random_chord = np.random.choice(self.chords)
        if random_chord.is_only_dim:
            return random_chord, random_chord.get_diminished()
        return random_chord, random_chord.get_notes()

    def get_population(self, population_size: int) -> List[Tuple[Chord, List[int]]]:
        """
        Population with contains the individuals
        :param population_size: number of individuals
        :return: List of tuples with chord and notes values of chord
        """
        return [self.get_individual() for _ in range(population_size)]

    def get_fitness(self,
                    melody_part: List[Union[str, int]],
                    prev_individual: Tuple[Chord, List[int]],
                    individual: Tuple[Chord, List[int]]
                    ) -> float:
        """
        Get fitness value for the individual
        :param melody_part: note values of melody part
        :param prev_individual: last chord in previous part of melody
        :param individual: current chord with sequence of note values
        :return: fintess value for current individual
        """
        count = 0
        key_val = Keys.names.index(self.key.replace('m', ''))
        dominant_value = (key_val + 7) % 12
        for note in melody_part:
            if note in individual[1]:
                count += 1
        if not (prev_individual is None):
            if not melody_part and individual[0].name == prev_individual[0].name and \
                    individual[1] == prev_individual[1]:
                count += 0.3
            if melody_part and individual[0].name != prev_individual[0].name:
                count += 0.2
        if 'm' in self.key:
            seventh_step = (key_val - 2) % 12
            sixth_step = (key_val + 8) % 12
        else:
            seventh_step = (key_val - 1) % 12
            sixth_step = (key_val + 9) % 12
        second_step = (key_val + 2) % 12
        forth_step = (key_val + 5) % 12

        if contains_sequence([seventh_step, second_step, key_val], melody_part) or \
                contains_sequence([second_step, seventh_step, key_val], melody_part):
            if individual[1] == individual[0].get_notes() and individual[0].name == self.key:
                count += 10
            if individual[1] == individual[0].get_notes() and \
                    individual[0].name.replace('m', '') == Keys.names[(key_val + 7) % 12]:
                count += 8

        elif contains_sequence([sixth_step, forth_step, dominant_value], melody_part) or \
                contains_sequence([forth_step, sixth_step, dominant_value], melody_part):
            if individual[1] == individual[0].get_notes() and \
                    individual[0].name.replace('m', '') == Keys.names[dominant_value]:
                count += 10

        if melody_part:
            if melody_part[0] == individual[1][0]:
                count += 0.1
            if melody_part[-1] == individual[1][0]:
                count += 0.1
            if contains_sequence(individual[1], individual[0].get_notes()):
                count += 0.2
        return count

    def population_fitness(self,
                           melody_part: List[Union[str, int]],
                           prev_individual: Tuple[Chord, List[int]],
                           population: List[Tuple[Chord, List[int]]]
                           ) -> List[float]:
        """
        Get fitness values of population
        :param melody_part: note values of melody part
        :param prev_individual: last chord in previous part of melody
        :param population: list of individuals (chords)
        :return: list of fitness values of population
        """
        fitness = [self.get_fitness(melody_part, prev_individual, individual) for individual in population]
        return fitness

    def crossover(self,
                  population: List[Tuple[Chord, List[int]]],
                  size: int
                  ) -> List[Tuple[Chord, List[int]]]:
        """
        Crossing two parents and producing two offspring
        :param population: list of individuals (chords)
        :param size: number of repeated crosses
        :return: offspring with represent the list of new individuals
        """
        offsprings = []
        for _ in range(size):
            first_parent = random.choice(population)
            second_parent = random.choice(population)
            min_ind = min(self.chords.index(first_parent[0]), self.chords.index(second_parent[0]))
            max_ind = max(self.chords.index(first_parent[0]), self.chords.index(second_parent[0]))
            first_offspring = self.chords[(max_ind - min_ind) // 2 + min_ind]
            offsprings.append((first_offspring, first_offspring.get_notes()))

            second_offspring = self.chords[((min_ind + 7 - max_ind) // 2 + max_ind) % 7]
            offsprings.append((second_offspring, second_offspring.get_notes()))

        return offsprings

    def mutation(self,
                 offsprings: List[Tuple[Chord, List[int]]],
                 size: int
                 ) -> List[Tuple[Chord, List[int]]]:
        """
        Offspring mutation
        :param offsprings: list of individuals (chords)
        :param size: number of repeated mutations
        :return: updated offspring
        """
        for _ in range(size):
            offspring = random.choice(offsprings)
            random_chord_type = random.choice(Chord.names[1:])
            if random_chord_type == "diminished":
                offsprings[offsprings.index(offspring)] = (offspring[0], offspring[0].get_diminished())
            elif random_chord_type == "sus2":
                if 'm' in self.key and self.chords.index(offspring[0]) + 1 not in [2, 5] or \
                        'm' not in self.key and self.chords.index(offspring[0]) + 1 not in [3, 7]:
                    offsprings[offsprings.index(offspring)] = (offspring[0], offspring[0].get_sus2())
            elif random_chord_type == "sus4":
                if 'm' in self.key and self.chords.index(offspring[0]) + 1 not in [2, 6] or \
                        'm' not in self.key and self.chords.index(offspring[0]) + 1 not in [4, 7]:
                    offsprings[offsprings.index(offspring)] = (offspring[0], offspring[0].get_sus4())

        return offsprings

    def selection(self,
                  population: List[Tuple[Chord, List[int]]],
                  population_fitness: List[float],
                  offsprings: List[Tuple[Chord, List[int]]],
                  offsprings_fitness: List[float],
                  size: int
                  ) -> List[Tuple[Chord, List[int]]]:
        """
        Selection of eligible individuals
        :param population: list of individuals (chords) which is parents for current generation
        :param population_fitness: value of fitness function for population
        :param offsprings: list of individuals (chords) which is offspring for current generation
        :param offsprings_fitness: value of fitness function for offspring
        :param size: number of individuals to take
        :return: new population
        """
        sort_index = np.argsort(population_fitness)
        population_sorted = []
        for i in sort_index:
            population_sorted.append(population[i])

        sort_index = np.argsort(offsprings_fitness)
        offsprings_sorted = []
        for i in sort_index:
            population_sorted.append(offsprings[i])

        parents = population_sorted[size:]
        offsprings = offsprings_sorted[-size:]

        return [*parents, *offsprings]

    def evolution(self,
                  melody_part: List[Union[str, int]],
                  prev_individual: Tuple[Chord, List[int]],
                  ) -> Tuple[Tuple[Chord, List[int]], Tuple[Chord, List[int]]]:
        """
        Evolution process for some melody part
        :param melody_part: note values of melody part
        :param prev_individual: last chord in previous part of melody
        :return: generated suitable chords for the current part of the melody
        """
        print(f"For the melody_part {melody_part}:")
        count_same_fitness = 0

        # generation of population
        population = self.get_population(self.population_size)

        # calculation the fitness for initial population
        fitness = self.population_fitness(melody_part, prev_individual, population)

        for generation in range(self.generations):
            prev_fitness = fitness
            offsprings = self.crossover(population, 10)  # crossing process
            offsprings = self.mutation(offsprings, 5)  # mutation process

            # calculation the fitness for offspring after crossing and mutation processes
            offsprings_fitness = self.population_fitness(melody_part, prev_individual, offsprings)
            population = self.selection(population, fitness, offsprings, offsprings_fitness, 3)  # selection process

            # calculation the fitness for new generated population after generation
            fitness = self.population_fitness(melody_part, prev_individual, population)

            if max(fitness) == max(prev_fitness):
                count_same_fitness += 1

            print(f"{generation + 1}. Chord {population[fitness.index(max(fitness))][0]} with fitness {max(fitness)}")

            # Condition of ending the evolution
            if count_same_fitness == EvolutionAlgorithm.COUNT_WITHOUT_CHANGING:
                break
            elif max(fitness) > 10:
                second_individual, fitness_second_individual = population[fitness.index(max(fitness))], max(fitness)
                fitness_copy = fitness.copy()
                fitness_copy.sort(reverse=True)
                fitness_first_individual = None
                for i in range(len(fitness_copy)):
                    if fitness_copy[i] != fitness_second_individual:
                        fitness_first_individual = fitness_copy[i]
                        break
                first_individual = population[fitness.index(fitness_first_individual)]
                print(
                    f"The first chord is {first_individual} with fitness {fitness_first_individual}\n"
                    f"The second chord is {second_individual} with fitness {fitness_second_individual}\n\n"
                )
                return first_individual, second_individual

        # Return the generated chords
        if max(fitness) == 0:
            return prev_individual, prev_individual
        else:
            individual, fitness_individual = population[fitness.index(max(fitness))], max(fitness)

            print(
                f"The first chord is {individual} with fitness {fitness_individual}\n"
                f"The second chord is {individual} with fitness {fitness_individual}\n\n"
            )
            return individual, individual


class AccompanimentGenerator:
    """
    Accompaniment Generator class
    """

    def __init__(self, path: str):
        """
        Accompaniment Generator class constructor
        :param path: path to file with initial melody
        """
        self.initial_melody = MidiFile(path, clip=True)
        self.accompaniment = MidiTrack()
        self.accompaniment.append(MetaMessage('track_name', name='Elec. Piano (Classic)', time=0))

    def divide_melody_by_parts(self) -> List[List[Union[str, int]]]:
        """
        Divide the all note values into the small parts (half of beat)
        :return: list of note values after division
        """
        time = 0
        notes_length = []
        for message in self.initial_melody.tracks[1][2:]:
            time += message.time
            if message.type == 'note_on':
                notes_length.append([message.note, time])
            elif message.type == 'note_off':
                notes_length[-1].append(time)

        notes = []
        num_parts = round(time / 768)
        for i in range(num_parts):
            part = []
            for note, start, finish in notes_length:
                for j in range(start, finish + 1):
                    if 768 * i < j < 768 * (i + 1):
                        part.append(note % 12)
                        break

            notes.append(part)
        return notes

    def include_accompaniment(self, chords: List[Tuple[Chord, List[int]]]) -> MidiTrack:
        """
        Combines melody and generated chords
        :param chords: generated chords
        :return: MidiTrack with accompaniment
        """
        base = 36
        time = 0
        chords_length = []
        for chord in chords:
            for i in range(len(chord)):
                if chord[1][i] == 0 or chord[1][i] == 1:
                    chord[1][i] += 12
            chords_length.append([chord[1], time, time + 384])
            time += 384

        time = 0
        cur_time = 0
        for message in self.initial_melody.tracks[-1][2:-1]:
            time += message.time
            while chords_length[0][2] <= time or chords_length[0][1] == 0:
                if chords_length[0][2] <= time:
                    self.accompaniment.append(Message('note_off', channel=0, note=base + chords_length[0][0][0],
                                                      velocity=50, time=chords_length[0][2] - cur_time))
                    self.accompaniment.append(Message('note_off', channel=0, note=base + chords_length[0][0][1],
                                                      velocity=50, time=0))
                    self.accompaniment.append(Message('note_off', channel=0, note=base + chords_length[0][0][2],
                                                      velocity=50, time=0))
                    cur_time = chords_length[0][2]
                    chords_length.pop(0)
                if not chords_length:
                    message.time = time - cur_time
                    self.accompaniment.append(message)
                    cur_time = time
                    break
                if chords_length[0][1] is not None and chords_length[0][1] <= time:
                    self.accompaniment.append(Message('note_on', channel=0, note=base + chords_length[0][0][0],
                                                      velocity=50, time=chords_length[0][1] - cur_time))
                    self.accompaniment.append(Message('note_on', channel=0, note=base + chords_length[0][0][1],
                                                      velocity=50, time=0))
                    self.accompaniment.append(Message('note_on', channel=0, note=base + chords_length[0][0][2],
                                                      velocity=50, time=0))
                    cur_time = chords_length[0][1]
                    chords_length[0][1] = None
            else:
                message.time = time - cur_time
                self.accompaniment.append(message)
                cur_time = time
        if chords_length:
            self.accompaniment.append(Message('note_off', channel=0, note=base + chords_length[0][0][0],
                                              velocity=50, time=chords_length[0][2] - cur_time))
            self.accompaniment.append(Message('note_off', channel=0, note=base + chords_length[0][0][1],
                                              velocity=50, time=0))
            self.accompaniment.append(Message('note_off', channel=0, note=base + chords_length[0][0][2],
                                              velocity=50, time=0))
            cur_time = chords_length[0][2]
        if time > cur_time:
            self.accompaniment.append(MetaMessage('end_of_track', time=time - cur_time))
        else:
            self.accompaniment.append(MetaMessage('end_of_track', time=0))
        return self.accompaniment

    def generate(self) -> MidiFile:
        """
        Method that generates a melody file with accompaniment from a source melody file
        :return: melody file with accompaniment
        """
        keys = Keys(self.initial_melody)
        melody_key, chords = keys.get_melody_key()
        print(f"Melody key is {melody_key} : {chords}")
        divided_melody = self.divide_melody_by_parts()
        evolutionAlgorithm = EvolutionAlgorithm(generations=30, population_size=10, chords=chords, key=melody_key)

        chords_for_accompaniment = []
        second_prev_individual = None
        for part in divided_melody:
            first_prev_individual, second_prev_individual = evolutionAlgorithm.evolution(part, second_prev_individual)
            chords_for_accompaniment.append(first_prev_individual)
            chords_for_accompaniment.append(second_prev_individual)

        print("Generated chords for accompaniment:")
        for chord, note_sequence in chords_for_accompaniment:
            print(f"{chord} -> {note_sequence}")

        only_accompaniment = self.include_accompaniment(chords_for_accompaniment)
        with_accompaniment = self.initial_melody
        with_accompaniment.tracks[-1] = only_accompaniment

        return with_accompaniment


try:
    melody_path = input("Enter the path to initial melody file (e.g. barbiegirl_mono.mid): ")  # path to initial melody file
    melody_with_accompaniment_path = "RafailVenediktovExample.mid"  # path to save the generated melody with accompaniment
    accompaniment = AccompanimentGenerator(melody_path).generate()  # executing the accompaniment generation process
    accompaniment.save(melody_with_accompaniment_path)  # saving the generated melody file with accompaniment
    print("\nEverything complite! ")
except Exception as e:
    print("Some problems with generation... Check the path to the initial melody file!")