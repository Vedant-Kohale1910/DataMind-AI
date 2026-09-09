from src.chroma_client import get_vectorstore


def load_retriever(k=5, fetch_k=15, search_type="similarity"):
    vectordb = get_vectorstore()

    if search_type == "mmr":
        search_kwargs = {"k": k, "fetch_k": fetch_k}
    else:
        search_kwargs = {"k": k}

    return vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )


def format_retrieved_chunks(docs):
    formatted = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        video_title = meta.get("video_title", "Unknown video")
        start = meta.get("start")
        end = meta.get("end")
        timestamp = f"{start:.1f}s–{end:.1f}s" if start is not None and end is not None else "N/A"

        formatted.append(
            f"[{i}] {video_title} ({timestamp})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


if __name__ == "__main__":
    print("Connecting to Chroma Cloud...\n")

    retriever = load_retriever(k=5)

    test_query = input("Enter query: ")
    results = retriever.invoke(test_query)

    print(f"\nQuery: {test_query}\n")
    print(format_retrieved_chunks(results))
