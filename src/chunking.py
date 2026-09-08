# Ran in Google Colab
import os
import json
import torch
from transformers import pipeline

audio_folder = "audios"
json_folder = "jsons"

os.makedirs(json_folder, exist_ok=True)

pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v2",
    torch_dtype=torch.float16,
    device=0
)

audios = sorted([f for f in os.listdir(audio_folder) if f.lower().endswith(".mp3")])

for audio in audios:
    audio_path = os.path.join(audio_folder, audio)
    filename = os.path.splitext(audio)[0]

    if ". " in filename:
        number, title = filename.split(". ", 1)
    else:
        number = filename
        title = filename

    json_path = os.path.join(json_folder, f"{number}.json")

    if os.path.exists(json_path):
        print(f"Skipping {audio}")
        continue

    print(f"\nProcessing: {audio}")

    result = pipe(
        audio_path,
        chunk_length_s=15,
        batch_size=1,
        return_timestamps=True,
        generate_kwargs={
            "task": "translate"
        }
    )

    chunks = []
    if "chunks" in result:
        for chunk in result["chunks"]:
            timestamp = chunk.get("timestamp", (None, None))
            start = timestamp[0]
            end = timestamp[1]
            chunks.append(
                {
                    "number": number,
                    "title": title,
                    "start": start,
                    "end": end,
                    "text": chunk["text"].strip()
                }
            )
    else:
        chunks.append(
            {
                "number": number,
                "title": title,
                "start": None,
                "end": None,
                "text": result["text"].strip()
            }
        )

    output = {
        "number": number,
        "title": title,
        "text": result["text"].strip(),
        "chunks": chunks
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved -> {json_path}")

print("\nAll files processed successfully!")