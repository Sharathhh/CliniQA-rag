from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load and Chunk the PDF 
loader = PyPDFLoader("document.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=45)
docs = splitter.split_documents(documents)

# Embedding Function 
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS Vector Store 
vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

# Load FLAN-T5 via Transformers Pipeline 
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Create RetrievalQA Chain 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Ask Questions 
query = 'Give me the correct coded classsification for the classification for the following diagnosis: Recurrent depressive, currently in remmission'
result = qa_chain(query)

# Print Output 
print("\n Answer:")
print(result["result"])


print("\n Retrieved Sources:\n")
for doc in result["source_documents"]:
    print(doc.page_content)
    print("-" * 80)
