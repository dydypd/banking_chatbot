from huggingface_hub import community

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import RecursiveCharacterTextSplitter as Rec
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_chroma import Chroma


def load_pdf_data():
    """
    Load the data from the PDF file
    :return: The data from the PDF file
    """
    pdf_loader = PyPDFLoader
    path = "./data/BIDV_Public_data.pdf"
    loader = pdf_loader(path)
    return loader.load()


def text_split(pdf):
    """
    Split the text into smaller chunks
    :param text: The text to split
    :return: The split text
    """
    text_spliter = Rec(chunk_size=1000, chunk_overlap=100)
    docs = text_spliter.split_documents(pdf)
    return docs


def vectorize_database(docs):
    """
    Vectorize the documents
    :param docs: The documents to vectorize
    :return: The vectorized documents
    """
    embeddings = HuggingFaceEmbeddings()

    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vector_db.as_retriever()
    return retriever


docs = text_split(load_pdf_data())

vectorize_database(docs)
model_pipeline = pipeline(
    "text - generation ",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto"
)

llm = HuggingFacePipeline(
    pipeline=model_pipeline,
)

prompt = hub.pull("rlm/rag -prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough
        ()}
        | prompt
        | llm
        | StrOutputParser()
)

USER_QUESTION = "BIDV là gì?"
output = rag_chain.invoke(USER_QUESTION)
answer = output.split('Answer :')[1].strip()
print(answer)
