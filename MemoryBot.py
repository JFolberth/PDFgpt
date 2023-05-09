# Import necessary modules
import re
import os
import time
from io import BytesIO
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
from langchain.memory import CosmosDBChatMessageHistory
from pypdf import PdfReader

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.formrecognizer import FormRecognizerClient

#setting up some global variable
deployment_name = 'gpt-4'
model = 'gpt-4'
embed_engine = 'text-embedding-ada-002'
SECTION_TO_EXCLUDE = ['title', 'sectionHeading', 'footnote', 'pageHeader', 'pageFooter', 'pageNumber']
PAGES_PER_EMBEDDINGS = 1
encoding_name ='cl100k_base'

#setting up the keys
openai.api_type = "azure"
openai.api_base = 
openai.api_version = "2023-03-15-preview"
openai.api_key = 
os.environ['OPENAI_API_KEY'] = openai.api_key = 

endpoint = 
key = 

cosmos_endpoint = 
cosmos_db = 
cosmos_container = 
cosmos_connection_string =
user_id = 'PDFgpt'
session_id = '01'
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
@st.cache_data
def test_embed():
    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = OpenAIEmbeddings(deployment=embed_engine,model=embed_engine, chunk_size=1)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)

    st.success("Embeddings done.", icon="✅")
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

cosmos = CosmosDBChatMessageHistory(
    cosmos_endpoint=cosmos_endpoint,
    cosmos_database=cosmos_db,
    cosmos_container=cosmos_container,
    connection_string=cosmos_connection_string,
    session_id=session_id,
    user_id=user_id,
    ttl=120
    )


# Set up the Streamlit app
st.title("🤖 Personalized PDF Bot with Memory 🧠 backed by Azure CosmosDB")
st.markdown(
    """ 
        ####  🗨️ Chat with your PDF files 📜 with `Conversational Buffer Memory backed by Azure CosmosDB`  
        > *powered by [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [AzureOpenAI]('https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/')*
        ----
    """
)

st.markdown(
    """
    `AzureOpenai`
    `AzureFormRecognizer`
    `AzureCosmosDB`
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

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])

if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)
    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]

        # Test the embeddings and save the index in a vector database
        index = test_embed()
        doc_chain = load_qa_with_sources_chain(llm=llm,chain_type="stuff",verbose=True)
        question_generator = LLMChain(llm=llm,prompt=CONDENSE_QUESTION_PROMPT)
        
        cosmos.prepare_cosmos()
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(
                memory_key='chat_history',return_messages=True,input_key='question',chat_memory=cosmos,k=20)
            

        # Set up the question-answering system

        qa = ConversationalRetrievalChain(
            retriever=index.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            memory=st.session_state.memory
            )

        # Allow the user to enter a query and generate a response
        query = st.text_input(
            "**What's on your mind?**",
            placeholder="Ask me anything from {}".format(name_of_file),
            )

        if query:
            with st.spinner(
                "Generating Answer to your Query : `{}` ".format(query)):
                #res = conversational_agent.run(query)
                chat_history = []
                res = qa.run({"question": query, "chat_history": chat_history})
                print(res)
                st.info(res, icon="🤖")

        # Allow the user to view the conversation history and other information stored in the agent's memory
        with st.expander("History/Memory"):
            st.session_state.memory
