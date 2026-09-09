import os
import json
import glob
from langchain_core.documents import Document
from chromadb.errors import ChromaError

from src.chroma_client import get_vectorstore, get_cloud_client, COLLECTION_NAME

# Chroma Cloud enforces a default per-request "number of records" quota on
# new accounts (commonly 300). Batches are kept safely under that so a
# fresh account doesn't hit "Quota exceeded" on the very first upload.
# If you've since requested a quota increase (see the link in any quota
# error message), you can raise this back up for faster uploads.
BATCH_SIZE = 200

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JSON_DIR = os.path.join(PROJECT_ROOT, "data", "processed_jsons")


def load_all_documents(json_dir):
    """
    Reads every video JSON in json_dir and returns (documents, ids).
    Each chunk gets a deterministic id of "{video_number}_{chunk_index}",
    so re-running this script never creates duplicates — it just
    upserts (overwrite-if-same-id, insert-if-new).
    """
    documents = []
    ids = []
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in '{json_dir}'")

    print(f"Found {len(json_files)} JSON files in '{json_dir}'")

    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            video_data = json.load(f)

        video_number = video_data.get("number")
        video_title = video_data.get("title")
        chunks = video_data.get("chunks", [])

        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "video_number": video_number,
                        "video_title": video_title,
                        "chunk_index": idx,
                        "start": chunk.get("start"),
                        "end": chunk.get("end"),
                        "source_file": os.path.basename(filepath),
                    },
                )
            )
            ids.append(f"{video_number}_{idx}")

    print(f"Loaded {len(documents)} total chunks from {len(json_files)} videos.")
    return documents, ids


def get_existing_ids():
    """
    Returns the set of chunk ids already present in the Chroma Cloud
    collection. Checked at the individual chunk level (not per-video),
    so a retry after a partial failure — e.g. a quota error mid-upload,
    like the one that just happened — correctly re-uploads only the
    specific chunks that never made it, rather than skipping an entire
    video just because a FEW of its chunks got in during an earlier
    successful batch. Returns an empty set if the collection doesn't
    exist yet (first run).
    """
    client = get_cloud_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return set()

    existing = collection.get(include=[])  # ids are always returned, no need to ask
    return set(existing["ids"])


def build_vectorstore(json_dir=DEFAULT_JSON_DIR, skip_existing=True, force=False):
    """
    Embeds every video JSON in json_dir and upserts it into Chroma Cloud.

    This is the function you re-run whenever you add new videos, or after
    a failed/interrupted upload:
      - Drop new video JSONs into data/processed_jsons/
      - Run: python -m src.embed_and_index
      - Only chunks NOT already in Chroma Cloud get embedded and uploaded
        (skip_existing=True, the default). This is checked per-chunk, so
        it's safe to re-run after any failure without creating duplicates
        or leaving a partially-uploaded video stuck half-indexed.

    Set force=True to re-embed and overwrite everything regardless of
    what's already indexed (useful if you changed the chunking strategy
    and want to rebuild from scratch).
    """
    documents, ids = load_all_documents(json_dir)

    if skip_existing and not force:
        existing_ids = get_existing_ids()
        if existing_ids:
            paired = list(zip(documents, ids))
            filtered = [
                (doc, doc_id) for doc, doc_id in paired
                if doc_id not in existing_ids
            ]
            skipped = len(documents) - len(filtered)
            if skipped:
                print(f"Skipping {skipped} chunks already indexed in Chroma Cloud.")
            if not filtered:
                print("Nothing new to index — everything is already in Chroma Cloud.")
                return get_vectorstore()
            documents, ids = (list(t) for t in zip(*filtered))

    vectordb = get_vectorstore()

    print(f"Uploading {len(documents)} chunks to Chroma Cloud collection '{COLLECTION_NAME}'...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        try:
            vectordb.add_documents(documents=batch_docs, ids=batch_ids)
        except ChromaError as e:
            if "Quota exceeded" in str(e):
                print(
                    f"\nQuota exceeded while uploading batch starting at chunk {i}.\n"
                    f"Chunks 0-{i} above were uploaded successfully and are safe — "
                    f"nothing is lost. Options:\n"
                    f"  1. Request a quota increase using the link in the error above, "
                    f"then simply re-run this script — it will pick up exactly where "
                    f"it left off (already-uploaded chunks are skipped automatically).\n"
                    f"  2. Lower BATCH_SIZE further at the top of this file and re-run.\n"
                )
            raise

        print(f"  Uploaded {min(i + BATCH_SIZE, len(documents))}/{len(documents)} chunks")

    print(f"\nDone. Chroma Cloud collection '{COLLECTION_NAME}' is up to date.")
    return vectordb


if __name__ == "__main__":
    build_vectorstore()
