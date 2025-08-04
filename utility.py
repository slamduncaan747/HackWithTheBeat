import mido
from collections import defaultdict


def midi_to_tokens(filename):
    mid = mido.MidiFile(filename)

    midi_info = {"ticks_per_beat": mid.ticks_per_beat, "type": mid.type, "metadata": {}}

    all_events = []

    for track_idx, track in enumerate(mid.tracks):
        current_time = 0
        track_metadata = []

        for msg in track:
            current_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                all_events.append((current_time, "note_on", msg.note, msg.velocity))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                all_events.append((current_time, "note_off", msg.note))
            else:
                track_metadata.append((current_time, msg))

        if track_metadata:
            midi_info["metadata"][track_idx] = track_metadata

    all_events.sort(key=lambda x: (x[0], x[1] == "note_on", x[2]))

    tokens = []
    last_time = 0

    all_deltas = set()
    for i in range(1, len(all_events)):
        delta = all_events[i][0] - all_events[i - 1][0]
        if delta > 0:
            all_deltas.add(delta)

    if all_deltas:
        import math

        min_time_unit = (
            math.gcd(*all_deltas) if len(all_deltas) > 1 else min(all_deltas)
        )
        min_time_unit = max(min_time_unit, 4)
    else:
        min_time_unit = 12

    midi_info["min_time_unit"] = min_time_unit

    for event in all_events:
        time = event[0]

        if time > last_time:
            delta_ticks = time - last_time
            time_units = delta_ticks // min_time_unit
            remainder = delta_ticks % min_time_unit

            if remainder >= min_time_unit // 2:
                time_units += 1

            if time_units > 0:
                while time_units > 0:
                    shift_amount = min(time_units, 64)
                    tokens.append(f"TIME_SHIFT_{shift_amount}")
                    time_units -= shift_amount
                last_time = time

        if event[1] == "note_on":
            note, velocity = event[2], event[3]
            vel_quantized = round(velocity / 4) * 4
            tokens.append(f"NOTE_ON_{note}_{vel_quantized}")
        else:
            tokens.append(f"NOTE_OFF_{event[2]}")

    return tokens, midi_info


def tokens_to_midi(tokens, midi_info=None):
    if midi_info is None:
        midi_info = {"ticks_per_beat": 480, "type": 1, "metadata": {}}

    midi_messages = []
    current_time = 0
    note_states = defaultdict(int)

    min_time_unit = midi_info.get("min_time_unit", 12)

    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            shift_value = int(token.split("_")[-1])
            current_time += shift_value * min_time_unit
        elif token.startswith("NOTE_ON_"):
            parts = token.split("_")
            note = int(parts[2])
            velocity = int(parts[3])
            note_states[note] = velocity
            msg = mido.Message(
                "note_on", note=note, velocity=velocity, time=current_time
            )
            midi_messages.append(msg)
            current_time = 0
        elif token.startswith("NOTE_OFF_"):
            parts = token.split("_")
            note = int(parts[2])
            if note in note_states:
                del note_states[note]
                msg = mido.Message("note_off", note=note, time=current_time)
                midi_messages.append(msg)
                current_time = 0

    return midi_messages


def create_midi_file(midi_messages, midi_info):
    new_mid = mido.MidiFile(
        type=midi_info["type"], ticks_per_beat=midi_info["ticks_per_beat"]
    )

    metadata = midi_info.get("metadata", {})

    if 0 in metadata:
        meta_track = mido.MidiTrack()
        for time, msg in metadata[0]:
            msg.time = time
            meta_track.append(msg)
        new_mid.tracks.append(meta_track)

    track = mido.MidiTrack()
    for msg in midi_messages:
        track.append(msg)

    track.append(mido.MetaMessage("end_of_track", time=0))
    new_mid.tracks.append(track)

    for track_idx in sorted(metadata.keys()):
        if track_idx == 0:
            continue
        meta_track = mido.MidiTrack()
        for time, msg in metadata[track_idx]:
            msg.time = time
            meta_track.append(msg)
        new_mid.tracks.append(meta_track)

    return new_mid
