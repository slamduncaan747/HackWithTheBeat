import os
import pickle
import numpy as np


def transpose_tokens(tokens, semitones):
    new_tokens = []
    for token in tokens:
        if token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_"):
            parts = token.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                note = int(parts[1])
                new_note = note + semitones
                if 0 <= new_note <= 127:
                    parts[1] = str(new_note)
                    new_tokens.append("_".join(parts))
                    continue
        new_tokens.append(token)
    return new_tokens


def apply_data_augmentation(all_tokens, augmentation_transposes=[-5, -3, 2, 4]):
    augmented_tokens = []

    for i, tokens in enumerate(all_tokens):
        if isinstance(tokens[0], int):
            print(f"failed, skipping sequence")
            augmented_tokens.append(tokens)
            continue

        augmented_tokens.append(tokens)

        for shift in augmentation_transposes:
            transposed = transpose_tokens(tokens, shift)
            augmented_tokens.append(transposed)

        if i % 10 == 0:
            print(f"  Processed {i+1}/{len(all_tokens)} sequences...")

    original_count = len(all_tokens)
    augmented_count = len(augmented_tokens)
    print(
        f"Data augmentation complete: {original_count} -> {augmented_count} sequences ({augmented_count/original_count:.1f}x)"
    )

    return augmented_tokens


def prepare_midi_dataset(use_augmentation=True):
    dataset_path = "../drive/MyDrive/dataset.pkl"

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    original_token_ids = dataset["token_ids"]
    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    vocab_size = dataset["vocab_size"]

    print(f"Dataset loaded successfully:")
    print(f"  Total tokens: {len(original_token_ids):,}")
    print(f"  Vocab size: {vocab_size:,}")

    # Apply data augmentation if requested
    if use_augmentation:
        original_tokens = [id_to_token[token_id] for token_id in original_token_ids]

        sequences = []
        current_sequence = []

        for token in original_tokens:
            if token == "<END>":
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = []
            else:
                current_sequence.append(token)

        if current_sequence:
            sequences.append(current_sequence)

        print(f"Found {len(sequences)} musical sequences for augmentation")

        augmented_sequences = apply_data_augmentation(sequences)

        all_tokens = []
        for sequence in augmented_sequences:
            all_tokens.extend(sequence)
            all_tokens.append("<END>")

        vocab = set(all_tokens)
        new_tokens = vocab - set(token_to_id.keys())
        if new_tokens:
            print(
                f"Warning: Found {len(new_tokens)} new tokens after augmentation: {new_tokens}"
            )
            for token in sorted(new_tokens):
                token_to_id[token] = len(token_to_id)
                id_to_token[len(id_to_token)] = token
            vocab_size = len(token_to_id)

        token_ids = [token_to_id[token] for token in all_tokens]

        print(f"Augmentation complete:")
        print(f"  Original tokens: {len(original_token_ids):,}")
        print(f"  Augmented tokens: {len(token_ids):,}")
    else:
        print("Skipping data augmentation")
        token_ids = original_token_ids

    data = np.array(token_ids, dtype=np.uint16)

    n = len(data)
    split_idx = int(n * 0.9)

    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train/val split:")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")

    output_dir = os.path.dirname(__file__)
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    train_data.tofile(train_path)
    val_data.tofile(val_path)

    print(f"Saved train.bin: {train_path}")
    print(f"Saved val.bin: {val_path}")

    meta = {
        "vocab_size": vocab_size,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
    }

    meta_path = os.path.join(output_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\nDataset statistics:")
    print(f"  Unique tokens in vocab: {len(set(token_ids))}")

    return {
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "vocab_size": vocab_size,
    }


if __name__ == "__main__":
    stats = prepare_midi_dataset()
    print(f"  Ready for training with {stats['train_tokens']:,} training tokens")
    print(f"  Validation set: {stats['val_tokens']:,} tokens")
    print(f"  Vocabulary size: {stats['vocab_size']:,}")
