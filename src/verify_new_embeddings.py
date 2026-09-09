"""
RUN THIS FIRST, before re-embedding anything.

Confirms HuggingFaceEndpointEmbeddings actually works with your token and
produces reasonable-looking embeddings (right dimension, non-zero values,
similar sentences score higher than dissimilar ones) BEFORE you overwrite
your entire production Chroma Cloud collection with it.

Usage: python -m src.verify_new_embeddings
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def main():
    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        print("ERROR: HF_API_TOKEN not set. Add it to your .env file first.")
        return

    print("Connecting to Hugging Face's hosted Inference API...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )

    test_sentences = [
        "Gradient descent is an optimization algorithm.",
        "Gradient descent minimizes a cost function iteratively.",  # similar to #1
        "The cat sat on the mat.",  # unrelated to #1 and #2
    ]

    print("Requesting embeddings for 3 test sentences...\n")
    vectors = embeddings.embed_documents(test_sentences)

    # --- Sanity check 1: dimension ---
    dim = len(vectors[0])
    print(f"Embedding dimension: {dim}")
    if dim != 384:
        print(
            f"  ⚠️  Expected 384 dimensions for all-MiniLM-L6-v2, got {dim}. "
            f"This suggests the hosted endpoint is NOT applying the same "
            f"pooling as local sentence-transformers. Do not proceed with "
            f"re-embedding until this is understood."
        )
    else:
        print("  ✅ Matches expected dimension for this model.")

    # --- Sanity check 2: not all zeros / not identical vectors ---
    all_same = all(v == vectors[0] for v in vectors)
    if all_same:
        print("  ⚠️  All three sentences produced IDENTICAL vectors — something is wrong.")
    else:
        print("  ✅ Vectors differ from each other as expected.")

    # --- Sanity check 3: similar sentences should score higher than unrelated ones ---
    sim_related = cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = cosine_similarity(vectors[0], vectors[2])

    print(f"\nCosine similarity (related sentences):   {sim_related:.4f}")
    print(f"Cosine similarity (unrelated sentences): {sim_unrelated:.4f}")

    if sim_related > sim_unrelated:
        print("  ✅ Related sentences score higher than unrelated ones — looks correct.")
        print("\nSafe to proceed with the full re-embed.")
    else:
        print("  ⚠️  Related sentences did NOT score higher than unrelated ones.")
        print("  This suggests the hosted endpoint's output is not comparable")
        print("  to what sentence-transformers computes locally. Do not run")
        print("  the full re-embed yet — something needs investigating first.")


if __name__ == "__main__":
    main()
