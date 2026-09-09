# Shared Chroma Cloud connection for the whole project.
#
# Both embed_and_index.py (writing) and retriever.py (reading) import from
# here, so there is exactly ONE place that knows how to connect to Chroma
# Cloud and ONE place that defines the collection name. This matters going
# forward: if you ever rename the collection or rotate credentials, you
# only change it here.
#
# Embeddings are generated via Hugging Face's hosted Inference API
# (HuggingFaceEndpointEmbeddings), NOT loaded locally. This is deliberate:
# loading sentence-transformers + torch in-process was what exceeded
# Render's free-tier 512MB RAM limit and crashed the deployed app. Calling
# a hosted API instead means torch never needs to be installed or loaded
# here at all.
#
# Requires four environment variables, set either in a local .env file
# (see .env.example) or as secrets on whatever platform hosts this app:
#     CHROMA_API_KEY
#     CHROMA_TENANT
#     CHROMA_DATABASE
#     HF_API_TOKEN


import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()  # loads .env locally; no-op in production if using host-level secrets

COLLECTION_NAME = "ml_ds_transcripts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_cloud_client = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        hf_token = os.environ.get("HF_API_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "Missing HF_API_TOKEN. Get a free token (Read access is enough) "
                "at huggingface.co/settings/tokens and set it in .env or as a "
                "platform secret."
            )
        _embeddings = HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
    return _embeddings


def get_cloud_client():
    global _cloud_client
    if _cloud_client is None:
        api_key = os.environ.get("CHROMA_API_KEY")
        tenant = os.environ.get("CHROMA_TENANT")
        database = os.environ.get("CHROMA_DATABASE")

        missing = [name for name, val in [
            ("CHROMA_API_KEY", api_key),
            ("CHROMA_TENANT", tenant),
            ("CHROMA_DATABASE", database),
        ] if not val]

        if missing:
            raise EnvironmentError(
                f"Missing Chroma Cloud credential(s): {', '.join(missing)}. "
                f"Set them in a local .env file (see .env.example) or as "
                f"secrets on your hosting platform."
            )

        _cloud_client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
    return _cloud_client


def get_vectorstore():
    client = get_cloud_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def get_cloud_client():
    global _cloud_client
    if _cloud_client is None:
        api_key = os.environ.get("CHROMA_API_KEY")
        tenant = os.environ.get("CHROMA_TENANT")
        database = os.environ.get("CHROMA_DATABASE")

        missing = [name for name, val in [
            ("CHROMA_API_KEY", api_key),
            ("CHROMA_TENANT", tenant),
            ("CHROMA_DATABASE", database),
        ] if not val]

        if missing:
            raise EnvironmentError(
                f"Missing Chroma Cloud credential(s): {', '.join(missing)}. "
                f"Set them in a local .env file (see .env.example) or as "
                f"secrets on your hosting platform."
            )

        _cloud_client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )
    return _cloud_client


def get_vectorstore():
    client = get_cloud_client()
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
