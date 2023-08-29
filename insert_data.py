#from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import DirectoryLoader, TextLoader

import os
#load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] ="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"

embeddings = OpenAIEmbeddings()
loader = DirectoryLoader(
    "./FAQ", glob="**/health.txt", loader_cls=TextLoader, show_progress=True
)
documents = loader.load()
#print(len(loader))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
docs = text_splitter.split_documents(documents)
#print(len(docs))
# PGVector needs the connection string to the database.
CONNECTION_STRING = "postgresql+psycopg2://admin:admin@127.0.0.1:5433/vectordb"
COLLECTION_NAME = "vectordb"
print('started')
PGVector.from_documents(
    docs,
    embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING
)
print('all good')