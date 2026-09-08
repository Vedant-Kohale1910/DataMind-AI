import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.retriever import load_retriever

# LLM_PROVIDER controls which backend generates answers:
#   "ollama" (default) -> your local Ollama install, for local dev/testing
#   "groq"             -> Groq's free hosted API, for the deployed/public app
#                         (same open-weight Llama model, just hosted remotely
#                         so it's reachable from a server that isn't your PC)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

OLLAMA_MODEL = "llama3.1:8b"
GROQ_MODEL = "openai/gpt-oss-20b"
temperature = 0.2

prompt_template = """You are a helpful ML/Data Science tutor.
Use the context below to answer the question. The context is made up of
multiple short excerpts from video transcripts — they may not individually
state the answer, but together they usually do. Synthesize a complete answer
by combining relevant information across all the excerpts provided.

Only say you don't have the information if the excerpts are genuinely
unrelated to the question — not simply because no single excerpt fully
answers it on its own.

DO NOT mention or cite source video titles, timestamps, or filenames in your
answer. Just provide a clear, direct explanation.

Context:
{context}

Question: {question}

Answer:"""


def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is not set. "
                "Get a free key at console.groq.com/keys."
            )
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, num_ctx=4096)


def format_docs(docs):
    formatted = []
    for d in docs:
        video_title = d.metadata.get("video_title", "Unknown video")
        start = d.metadata.get("start")
        end = d.metadata.get("end")
        timestamp = f" ({start:.1f}s–{end:.1f}s)" if start is not None and end is not None else ""

        formatted.append(f"[Source: {video_title}{timestamp}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(k=5):
    retriever = load_retriever(k=k)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_answer_with_sources(question, k=5):
    retriever = load_retriever(k=k)
    docs = retriever.invoke(question)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(prompt_template)

    context = format_docs(docs)
    formatted_prompt = prompt.invoke({"context": context, "question": question})
    answer = llm.invoke(formatted_prompt)

    return {
        "answer": StrOutputParser().invoke(answer),
        "sources": docs
    }


if __name__ == "__main__":
    print(f"Using LLM_PROVIDER={LLM_PROVIDER}\n")
    chain = build_rag_chain()
    test_question = input("Enter question: ")
    print(f"Question: {test_question}\n")
    print(chain.invoke(test_question))
