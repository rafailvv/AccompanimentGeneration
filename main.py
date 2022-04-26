from mido import MidiFile, MidiTrack, second2tick, Message

mid = MidiFile('new_song.mid', clip=True)
mid.type = 1
print(mid)

accompaniment = MidiTrack()
accompaniment.append(Message('note_on', channel=0, note=68, velocity=127, time=0))

mid.save('new_song.mid')