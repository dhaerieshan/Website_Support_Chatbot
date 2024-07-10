from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from pinecone import Pinecone as PineconeClient
import asyncio
from langchain.document_loaders.sitemap import SitemapLoader


def get_website_data(sitemap_url):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(
    sitemap_url
    )

    docs = loader.load()

    return docs

def split_data(docs):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

def create_embeddings():

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    PineconeClient(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name
    index =  Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    PineconeClient(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def get_similar_docs(index,query,k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


