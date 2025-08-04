import os
import pickle
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


def transpose_tokens(tokens, semitones):
    new_tokens = []
    for token in tokens:
        if token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_"):
            parts = token.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                note = int(parts[1])
                new_note = note + semitones
                if 0 <= new_note <= 127:
                    if token.startswith("NOTE_ON_") and len(parts) >= 3:
                        parts[1] = str(new_note)
                        new_tokens.append("_".join(parts))
                    else:
                        parts[1] = str(new_note)
                        new_tokens.append("_".join(parts))
                    continue
        new_tokens.append(token)
    return new_tokens


def build_dataset(midi_folder, output_file="dataset.pkl", use_augmentation=True):
    all_tokens = []
    vocab = set()
    total_files = 0
    processed_files = 0

    augmentation_transposes = [-5, -3, 2, 4] if use_augmentation else []

    midi_files = []

    for root, dirs, files in os.walk(midi_folder):
        for filename in files:
            if filename.endswith(".mid") or filename.endswith(".midi"):
                midi_files.append(os.path.join(root, filename))

    total_files = len(midi_files)

    for filepath in midi_files:
        filename = os.path.basename(filepath)
        relative_path = os.path.relpath(filepath, midi_folder)

        try:
            tokens, midi_info = midi_to_tokens(filepath)

            all_tokens.append(tokens)
            vocab.update(tokens)

            if use_augmentation:
                for semitones in augmentation_transposes:
                    transposed_tokens = transpose_tokens(tokens, semitones)
                    all_tokens.append(transposed_tokens)
                    vocab.update(transposed_tokens)

            processed_files += 1
            multiplier = len(augmentation_transposes) + 1 if use_augmentation else 1
            print(
                f"Processed ({processed_files}/{total_files}) {relative_path}: {len(tokens)} tokens × {multiplier} versions"
            )
        except Exception as e:
            print(f"Failed on {relative_path}: {e}")

    vocab = sorted(list(vocab))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    token_to_id["<PAD>"] = len(token_to_id)
    token_to_id["<END>"] = len(token_to_id)

    all_ids = []
    for tokens in all_tokens:
        ids = [token_to_id[t] for t in tokens]
        all_ids.extend(ids)
        all_ids.append(token_to_id["<END>"])

    dataset = {
        "token_ids": all_ids,
        "token_to_id": token_to_id,
        "id_to_token": {i: t for t, i in token_to_id.items()},
        "vocab_size": len(token_to_id),
    }

    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset stats:")
    print(f"Total tokens: {len(all_ids):,}")
    print(f"Vocab size: {len(token_to_id):,}")
    print(f"Files processed: {processed_files}/{total_files}")

    if use_augmentation:
        original_versions = processed_files
        augmented_versions = len(all_tokens)
        augmentation_factor = (
            augmented_versions / original_versions if original_versions > 0 else 0
        )
        print(
            f"🎵 Data augmentation: {original_versions} original → {augmented_versions} total versions ({augmentation_factor:.1f}x)"
        )

    if all_tokens:
        token_counts = [len(tokens) for tokens in all_tokens]
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"Average tokens per version: {avg_tokens:.0f}")
        print(f"Largest version: {max(token_counts):,} tokens")
        print(f"Smallest version: {min(token_counts):,} tokens")

    return dataset


dataset = build_dataset(
    "adl-piano-midi/midi/adl-piano-midi", "dataset.pkl", use_augmentation=True
)
