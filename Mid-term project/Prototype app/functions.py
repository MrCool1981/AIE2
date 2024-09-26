from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
import tiktoken
import os

### SETUP FUNCTIONS ###
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

def setup_vector_db():

    # Get the directory of the current file
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the current file's directory
    os.chdir(current_file_directory)

    # Load the NIST AI document
    PDF_LINK = "data/nist_ai.pdf"
    loader = PyMuPDFLoader(file_path=PDF_LINK)
    nist_doc = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = tiktoken_len,
    )

    nist_chunks = text_splitter.split_documents(nist_doc)

    embeddings_small = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-small")

    qdrant_client = QdrantClient(":memory:") # set Qdrant DB and its location (in-memory)

    qdrant_client.create_collection(
        collection_name="NIST_AI",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="NIST_AI",
        embedding=embeddings_small,
    ) # create a QdrantVectorStore object with the above specified client, collection name, and embedding model.

    qdrant_vector_store.add_documents(nist_chunks) # add the documents to the QdrantVectorStore

    retriever = qdrant_vector_store.as_retriever()

    return retriever

### VARIABLES ###

# define a global variable to store the retriever object
retriever = setup_vector_db()
qa_gpt4_llm = AzureChatOpenAI(azure_deployment="gpt-4", temperature=0) # GPT-4o model

# define a template for the RAG model
rag_template = """
You are a helpful assistant that helps users find information and answer their question. 
You MUST use ONLY the available context to answer the question.
If necessary information to answer the question cannot be found in the provided context, you MUST "I don't know."

Question:
{question}

Context:
{context}
"""
# create rag prompt object from the template
prompt = ChatPromptTemplate.from_template(rag_template)

# update the chain with LLM, prompt, and question variable.
retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | qa_gpt4_llm, "context": itemgetter("context"), "question": itemgetter("question")}
)

### FUNCTIONS ###


def get_response(query, history):
    """A helper function to get the response from the RAG model and return it to the UI."""

    response = retrieval_augmented_qa_chain.invoke({"question" : query})
    
    return response["response"].content