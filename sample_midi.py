import os
import pickle
import torch
import numpy as np
from contextlib import nullcontext
import mido
from collections import defaultdict

from model import GPTConfig, GPT

import sys

sys.path.append(".")
from utility import midi_to_tokens, tokens_to_midi, create_midi_file


out_dir = "out-midi"
start = None
num_samples = 3
max_new_tokens = 1000
seed = 2500
device = "cuda"
dtype = "float32"
compile = True

temperature_stage1 = 0.95
temperature_stage2 = 0.75
top_k_stage1 = 10
top_k_stage2 = 25
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


def load_model():
    ckpt_path = "../drive/MyDrive/music_model_150M/best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if compile:
        model = torch.compile(model)

    meta_path = os.path.join("data", "midi", "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return model, meta


def get_starting_sequence(model, length=256):
    data_path = os.path.join("data", "midi", "train.bin")
    data = torch.from_numpy(np.memmap(data_path, dtype=np.uint16, mode="r")).long()

    max_start = len(data) - length
    if max_start <= 0:
        return data.tolist()

    start_idx = torch.randint(0, max_start, (1,)).item()
    sequence = data[start_idx : start_idx + length]

    return sequence.tolist()


def generate_tokens(
    model,
    meta,
    start_tokens,
    max_new_tokens,
    temperature_stage1=1.0,
    temperature_stage2=1.0,
    top_k_stage1=50,
    top_k_stage2=10,
):
    model.eval()

    if isinstance(start_tokens, str):
        if start_tokens in meta["token_to_id"]:
            start_ids = [meta["token_to_id"][start_tokens]]
        else:
            start_ids = [0]
    else:
        start_ids = start_tokens

    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    block_size = model.config.block_size
    token_groups = defaultdict(list)
    base_token_names = {}

    for token_str, token_id in meta["token_to_id"].items():
        if token_str.startswith("NOTE_ON_") or token_str.startswith("NOTE_OFF_"):
            parts = token_str.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                note_num = int(parts[2])
                base_token = f"NOTE_{note_num}"
                token_groups[base_token].append(token_id)
                base_token_names[base_token] = f"Note {note_num}"
        elif token_str.startswith("TIME_SHIFT_"):
            base_token = token_str
            token_groups[base_token].append(token_id)
            base_token_names[base_token] = token_str
        else:
            base_token = token_str
            token_groups[base_token].append(token_id)
            base_token_names[base_token] = token_str

    with torch.no_grad():
        with ctx:
            for k in range(max_new_tokens):
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

                logits, _ = model(x_cond)
                base_logits = logits[:, -1, :]

                base_token_scores = {}
                for base_token, token_ids in token_groups.items():
                    score = 0.0
                    for tid in token_ids:
                        if tid < base_logits.size(-1):
                            score += torch.exp(
                                base_logits[0, tid] / temperature_stage1
                            ).item()
                    if score > 0:
                        base_token_scores[base_token] = score

                if not base_token_scores:
                    print("Warning: No valid base tokens found")
                    break

                sorted_base_tokens = sorted(
                    base_token_scores.items(), key=lambda x: x[1], reverse=True
                )
                if top_k_stage1 is not None and len(sorted_base_tokens) > top_k_stage1:
                    sorted_base_tokens = sorted_base_tokens[:top_k_stage1]

                base_token_list = [bt for bt, _ in sorted_base_tokens]
                base_token_probs = torch.tensor(
                    [score for _, score in sorted_base_tokens], device=device
                )
                base_token_probs = base_token_probs / base_token_probs.sum()

                sampled_idx = torch.multinomial(base_token_probs, num_samples=1).item()
                selected_base_token = base_token_list[sampled_idx]

                candidate_token_ids = token_groups[selected_base_token]

                if len(candidate_token_ids) == 1:
                    next_token = torch.tensor([[candidate_token_ids[0]]], device=device)
                else:
                    candidate_logits = torch.tensor(
                        [
                            base_logits[0, tid].item()
                            for tid in candidate_token_ids
                            if tid < base_logits.size(-1)
                        ],
                        device=device,
                    )
                    candidate_logits = candidate_logits / temperature_stage2

                    if (
                        top_k_stage2 is not None
                        and len(candidate_logits) > top_k_stage2
                    ):
                        topk_values, topk_indices = torch.topk(
                            candidate_logits, top_k_stage2
                        )
                        filtered_token_ids = [
                            candidate_token_ids[i] for i in topk_indices.tolist()
                        ]
                        filtered_logits = topk_values
                    else:
                        filtered_token_ids = candidate_token_ids
                        filtered_logits = candidate_logits

                    candidate_probs = torch.nn.functional.softmax(
                        filtered_logits, dim=-1
                    )
                    sampled_variant_idx = torch.multinomial(
                        candidate_probs, num_samples=1
                    ).item()
                    next_token = torch.tensor(
                        [[filtered_token_ids[sampled_variant_idx]]], device=device
                    )

                x = torch.cat((x, next_token), dim=1)

                if k % 100 == 0:
                    token_str = meta["id_to_token"].get(next_token.item(), "UNKNOWN")
                    print(f"Step {k}: {selected_base_token} -> {token_str}")

                if next_token.item() == meta["token_to_id"].get("<END>", -1):
                    break

    return x[0].tolist()


def tokens_to_midi_file(tokens, meta, output_filename, default_midi_info=None):
    token_strings = []
    for token_id in tokens:
        if token_id in meta["id_to_token"]:
            token_strings.append(meta["id_to_token"][token_id])
        else:
            print(f"Warning: Unknown token ID {token_id}")

    filtered_tokens = []
    for token in token_strings:
        if token.startswith("<"):
            continue

        if token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_"):
            try:
                parts = token.split("_")
                note_num = int(parts[2])

                if note_num < 0 or note_num > 127:
                    print(
                        f"Warning: Invalid MIDI note {note_num} in token '{token}', skipping"
                    )
                    continue

                if token.startswith("NOTE_ON_") and len(parts) > 3:
                    velocity = int(parts[3])
                    if velocity < 0 or velocity > 127:
                        print(
                            f"Warning: Invalid MIDI velocity {velocity} in token '{token}', clamping to 0-127"
                        )
                        velocity = max(0, min(127, velocity))
                        parts[3] = str(velocity)
                        token = "_".join(parts)

            except (ValueError, IndexError):
                print(f"Warning: Malformed token '{token}', skipping")
                continue

        filtered_tokens.append(token)

    if not filtered_tokens:
        print("Warning: No valid music tokens generated after filtering")
        return False

    if default_midi_info is None:
        default_midi_info = {
            "ticks_per_beat": 384,
            "type": 1,
            "metadata": {},
            "min_time_unit": 4,
        }

    try:
        midi_messages = tokens_to_midi(filtered_tokens, default_midi_info)

        new_mid = create_midi_file(midi_messages, default_midi_info)
        new_mid.save(output_filename)

        print(f"Generated MIDI file: {output_filename}")
        print(f"  Tokens used: {len(filtered_tokens)}")
        return True

    except Exception as e:
        print(f"Error converting tokens to MIDI: {e}")
        return False


def main():
    model, meta = load_model()

    os.makedirs("generated_midi", exist_ok=True)

    for i in range(num_samples):
        print(f"\n{'='*60}")
        print(f"Generating sample {i+1}/{num_samples}...")

        start_tokens = get_starting_sequence(model, length=256)

        full_tokens = generate_tokens(
            model,
            meta,
            start_tokens,
            max_new_tokens,
            temperature_stage1=temperature_stage1,
            temperature_stage2=temperature_stage2,
            top_k_stage1=top_k_stage1,
            top_k_stage2=top_k_stage2,
        )

        print(
            f"\nGenerated {len(full_tokens)} total tokens ({len(full_tokens) - len(start_tokens)} new)"
        )

        output_file = f"generated_midi/generated_bach_2stage_{i+1}.mid"
        success = tokens_to_midi_file(full_tokens, meta, output_file)

        if success:
            print(f"✓ Successfully generated: {output_file}")
            print(f"  Total length: {len(full_tokens)} tokens")
        else:
            print(f"✗ Failed to generate: {output_file}")


if __name__ == "__main__":
    main()
