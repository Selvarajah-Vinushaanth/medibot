from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbotnew"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
