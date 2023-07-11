# Import necessary modules
import re
import os
from io import BytesIO
import json
from typing import Any, Dict, List

import openai
import streamlit as st
from langchain import LLMChain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain,ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import OutputFixingParser
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ChatMessageHistory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import ConversationalRetrievalChain
from pypdf import PdfReader

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.formrecognizer import FormRecognizerClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

#setting up some global variable
deployment_name = 'gpt-35-turbo'
model = 'gpt-35-turbo'
embed_engine = 'text-embedding-ada-002'
SECTION_TO_EXCLUDE = ['title', 'sectionHeading', 'footnote', 'pageHeader', 'pageFooter', 'pageNumber']
PAGES_PER_EMBEDDINGS = 1
encoding_name ='cl100k_base'

def check_config_data(config_data):
    if not "storage_container" in config_data:
        return False
    if not "key_vault_name" in config_data:
        return False
    return True
def read_config_data(filename):
    config_data = json.loads(open(filename).read())
    if not check_config_data(config_data):
        print("Config data is not valid")
        raise Exception("Config data is not valid")
    return config_data
config_data = read_config_data("app/app_config.json")
KVUri = f"https://{config_data['key_vault_name']}.vault.azure.net"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KVUri, credential=credential)
account_url= client.get_secret('sa-endpoint').value
#setting up the keys
openai.api_type = "azure"
openai.api_base = client.get_secret('openai-endpoint').value
openai.api_version = "2023-03-15-preview"
openai.api_key = client.get_secret('openai-key').value
os.environ['OPENAI_API_KEY'] = openai.api_key 
os.environ['OPENAI_API_BASE'] = openai.api_base
os.environ['OPENAI_API_VERSION'] = openai.api_version 
os.environ['OPENAI_API_TYPE'] = openai.api_type
default_credential = DefaultAzureCredential()


endpoint = client.get_secret('form-recognizer-endpoint').value
key = client.get_secret('form-recognizer-key').value


def process_pdf(data):
    doc = parse_pdf(data)
    pages = text_to_docs(doc)
    return pages



def list_blobs(blob_service_client: BlobServiceClient, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()
    return blob_list

def upload_blob_data(blob_service_client: BlobServiceClient, container_name, data, filename):
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    # Upload the blob data - default blob type is BlockBlob
    blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=True)

def check_process_file(current_filename):
    if not ".pdf" in current_filename:
        return False
    return True

def get_blob_service_client_connection_string(account_url):
    
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    return blob_service_client

def load_FAISS_vector_store(config_data):
    input_container_name = config_data["storage_container"]
    blob_service_client = get_blob_service_client_connection_string(account_url)
    blob_list = list_blobs(blob_service_client, input_container_name)
    first_run = True
    index = None
    embeddings = OpenAIEmbeddings(deployment=embed_engine,model=embed_engine, chunk_size=1)
    for blob in blob_list:
        if check_process_file(blob.name):
            blob_client = blob_service_client.get_blob_client(container=input_container_name, blob=blob.name)
            stream = blob_client.download_blob()
            data = stream.readall()
            pages = process_pdf(data)
            if first_run:
                index = initialize_embed(pages, embeddings)
                first_run = False
            else:
                index2 = initialize_embed(pages, embeddings)
                index.merge_from(index2)
                
    return index
#user_id = 'PDFgpt'
#session_id = '01'
# Normalizing text to remove \n and other seperators
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

# Define a function to parse a PDF file and extract its text content
@st.cache_data

def parse_pdf(doc):

    # Extract the pdf contenct

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )
    
    
    poller_layout = document_analysis_client.begin_analyze_document(
            model_id = "prebuilt-document", document = doc)
    layout = poller_layout.result()

    results = [""] * (len(layout.pages) // PAGES_PER_EMBEDDINGS + 1)  # Initialize results list with empty strings

    for p in layout.paragraphs:
        page_number = p.bounding_regions[0].page_number
        output_file_id = int((page_number - 1 ) / PAGES_PER_EMBEDDINGS)

        if len(results) < output_file_id + 1:
            results.append('')

        if p.role not in SECTION_TO_EXCLUDE:
            results[output_file_id] += f"{p.content}\n"

    for t in layout.tables:
        page_number = t.bounding_regions[0].page_number
        output_file_id = int((page_number - 1 ) / PAGES_PER_EMBEDDINGS)
        
        if len(results) < output_file_id + 1:
            results.append('')
        previous_cell_row=0
        rowcontent='| '
        tablecontent = ''
        for c in t.cells:
            if c.row_index == previous_cell_row:
                rowcontent +=  c.content + " | "
            else:
                tablecontent += rowcontent + "\n"
                rowcontent='|'
                rowcontent += c.content + " | "
                previous_cell_row += 1
        results[output_file_id] += f"{tablecontent}|"

    return results


# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: List) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""

    page_docs = [Document(page_content=normalize_text(page)) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
            length_function =len
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# Define a function for the embeddings
def initialize_embed(pages, embeddings):
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)

    
    return index

#initializing chat instance from langchain
llm = AzureChatOpenAI(
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    deployment_name=deployment_name,
    openai_api_key=openai.api_key,
    openai_api_type=openai.api_type,
    model_name=model,
    verbose=True, 
    temperature=0
    )



# Set up the Streamlit app
st.title("🤖 Personalized PDF Bot ")
st.markdown(
    """ 
        ####  🗨️ Chat with your PDF files 📜 
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [AzureOpenAI]('https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/')*
        ----
    """
)

st.markdown(
    """
    `AzureOpenai`
    `AzureFormRecognizer`
    `langchain`
    `faiss-cpu`
    
    ---------
    """
)

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    3. Perform Q&A
    """
)


index = load_FAISS_vector_store(config_data)
st.success("Initial embeddings done.", icon="✅")


 
# Set up the question-answering system
doc_chain = load_qa_with_sources_chain(llm=llm,chain_type="stuff",verbose=True)
question_generator = LLMChain(llm=llm,prompt=CONDENSE_QUESTION_PROMPT)
qa = ConversationalRetrievalChain(
retriever=index.as_retriever(),
question_generator=question_generator,
combine_docs_chain=doc_chain

)

# Allow the user to enter a query and generate a response
query = st.text_input(
"**What's on your mind?**",
placeholder="Ask me anything from your data"
)

if query:
    with st.spinner(
        "Generating Answer to your Query : `{}` ".format(query)):
        #res = conversational_agent.run(query)
        chat_history = []
        res = qa.run({"question": query, "chat_history": chat_history})
        print(res)
        st.info(res, icon="🤖")

# Allow the user to upload a PDF file
# uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])
uploaded_file = None
if not uploaded_file == None:
    name_of_file = uploaded_file.name
    pages = process_pdf(uploaded_file)
    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]
        embeddings = OpenAIEmbeddings(deployment=embed_engine,model=embed_engine, chunk_size=1)
        # Test the embeddings and save the index in a vector database
        index2 = initialize_embed(pages, embeddings)
        index.merge_from(index2)
        doc_chain = load_qa_with_sources_chain(llm=llm,chain_type="stuff",verbose=True)
        question_generator = LLMChain(llm=llm,prompt=CONDENSE_QUESTION_PROMPT)
       


