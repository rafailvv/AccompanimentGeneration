# Descriptions of the Accompaniment Creation Process

The process was divided into the following main steps:

• Reading the notes of a melody from a `MIDI` file

• Determining the key of a melody

• Chord generation using evolution algorithm

• Adding chords to the initial melody

Some of the steps are described in more detail below. All process
located in the `AccopmpanimentGenerator` class and performed using
the `generate` method.


## Reading the notes of a melody from a MIDI file


The `mido` library is used to work with `*.mid` files in Python. The file
is opened, all the notes of the melody are read and they are divided
into groups. All this is implemented in the `divide_melody_by_parts`
method


## Determining the key of a melody

The algorithm is located in the `Keys` class. The constructor
generates a set of all possible major and minor keys with chords that
applicable in each key based on the Figure 6 and 7 in Assignment
description.

Determination of key consists of the following criteria:

- The number of matching notes. A list of keys that have the largest
number of matching notes is determined;

- Among the selected keys, the first and last notes in the melody
are analyzed (1 point for the tonic, 0.66 for the dominant and
subdominant, 0.33 for the third degree);

- In addition, the number of stable notes in the melody for each key
is determined. The key with the most notes gets 1 more point.

The key with the most points is the key of the given melody.

The `get_melody_key` method returns the key name and a list of the
names of all chords in the computed key.

## Chord generation using evolution algorithm

To generate the accompaniment, I used a type of evolutionary
algorithm - **genetic algorithm**. The algorithm is located in the
`EvolutionAlgorithm` class. Evolution occurs for each part separately.

The **genotype** is a musical chord consisting of three notes and one
of the following types: major and minor triads, first and second
inversions of major and minor triads, diminished chords (DIM),
suspended second chords (SUS2), suspended fourth chords (SUS4).

A **population** is a set of individuals (chords) included in the key of
a melody.

The **fitness** function for the entire population is the generation of
a list of fitness values for each individual (chord). It’s implemented in
the `get_fitness` method Fitness values depend on the following criteria:
- The number of notes in a certain part of the melody that are
contained in the chord (1 point per match);
- 0.3 points if the melody part is empty and the individual matches the
individual from the previous melody part;
- 0.2 points if the melody part is not empty and the individual does
NOT match the individual from the previous melody part;
- If a resolution to the tonic is found (enumeration of the seventh and
second degrees, followed by the tonic) 10 points if the individual is
the tonic and 8 points if the individual is the dominant;
- If a resolution to a dominant is found (search of the fourth and a
procession of steps, followed by a dominant), 10 points if the
individual is a dominant;
- 0.1 point if the root of the chord is the same as the first note in the
melody part;
- 0.1 point if the root of the chord is the same as the last note in the
melody part;
- 0.2 points if chord type is triad.


**Crossover** is a point crossing, which is implemented in the
`crossover` method and occurs as follows:
- Two parents (chords) are randomly selected from the population;
- As a result of crossing, two children (triad chords) are created, which
are the middle between the parents. *For example, in the key of C,
two parents C and F were chosen. Therefore, the offspring will be A
and D;*
- For the population, the crossing process is repeated 10 times.

**Mutation** is implemented in the `mutation` method and represent
the process of choosing a random representative of the offspring and
changing its chord type to DIM, SUS2 or SUS4. For the population, the
mutation process is repeated 5 times.

**Selecting** is implemented in the `selection` method and represent
the process of sorting fitness for population and offspring. Three worst
individuals are removed from the population and three individuals with
the highest fitness value are added, which eventually form a new
population.

The **process of evolution** is the generation of a field of 30
individuals for a particular part of a musical melody. Then, for 10
generations, the process of calculating the fitness function, crossing,
mutating and selecting suitable individuals for a new population takes
place. The result of the output is a repetition of the maximum fitness
value 30 times without change or finding a fitness value greater or
equal than 10, which demonstrates resolution to the tonic or dominant
in the melody.

The result of evolution is two individuals (chords with a certain
type) that have the highest fitness value.

# Notes of output melodies

![image](https://user-images.githubusercontent.com/69470761/178219518-686707ee-306e-4ce8-aafa-f2b414a21237.png)

![image](https://user-images.githubusercontent.com/69470761/178219632-ca55c5ea-7d88-46c5-b517-b6aa08c9ab1d.png)

![image](https://user-images.githubusercontent.com/69470761/178219780-2ffe635e-64f2-4c75-b965-ae87c705b6ed.png)




