#from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader


import os
import pandas as pd
#load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()
# loader = DirectoryLoader(
#     "./FAQ", glob="**/health.txt", loader_cls=TextLoader, show_progress=True
# )
# documents = loader.load()
#print(len(loader))

loader = CSVLoader(file_path='./service3/input_data.csv')
data = loader.load()
data=data[0:1000]
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=250)
docs = text_splitter.split_documents(data)
print(len(docs))
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
#df=pd.read_csv('./service3/input_data.csv')
#print(df.shape)
print('all good')