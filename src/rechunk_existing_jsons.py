import os
import json
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_jsons")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_jsons_fixed")
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def build_full_text_with_offsets(raw_segments):
    full_text = ""
    offset_map = []

    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        char_start = len(full_text)
        full_text += (" " if full_text else "") + text
        char_end = len(full_text)
        offset_map.append((char_start, char_end, seg["start"], seg["end"]))

    return full_text, offset_map

def find_chunk_timestamps(chunk_text, search_start, full_text, offset_map):
    pos = full_text.find(chunk_text, max(0, search_start - CHUNK_OVERLAP))
    if pos == -1:
        pos = full_text.find(chunk_text)
    chunk_start_char = pos
    chunk_end_char = pos + len(chunk_text)

    overlapping = [
        seg for seg in offset_map
        if seg[1] > chunk_start_char and seg[0] < chunk_end_char
    ]

    if overlapping:
        start_time = overlapping[0][2]
        end_time = overlapping[-1][3]
    else:
        start_time, end_time = None, None

    return start_time, end_time, chunk_end_char

def reprocess_all():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"Reading from: {INPUT_DIR}")
    print(f"Writing to:   {OUTPUT_DIR}")
    print(f"Found {len(input_files)} JSON files\n")

    if not input_files:
        print("No files found. Double-check INPUT_DIR actually contains your 134 JSONs.")
        return

    for filepath in input_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        number = data.get("number")
        title = data.get("title")
        raw_segments = data.get("chunks", [])

        full_text, offset_map = build_full_text_with_offsets(raw_segments)
        split_texts = splitter.split_text(full_text)

        new_chunks = []
        cursor = 0
        for chunk_text in split_texts:
            start_time, end_time, new_cursor = find_chunk_timestamps(
                chunk_text, cursor, full_text, offset_map
            )
            cursor = new_cursor

            new_chunks.append({
                "number": number,
                "title": title,
                "start": start_time,
                "end": end_time,
                "text": chunk_text
            })

        output_data = {
            "number": number,
            "title": title,
            "text": data.get("text", ""),
            "chunks": new_chunks
        }

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"  {os.path.basename(filepath)}: {len(raw_segments)} segments -> {len(new_chunks)} chunks")

    print(f"\nDone. Fixed files written to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    reprocess_all()