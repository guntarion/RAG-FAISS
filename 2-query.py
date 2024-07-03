# query.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set your OpenAI API key
load_dotenv()

def load_qa_chain():
    # 1. Load the saved FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # 2. Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

def query_data(qa_chain, query):
    response = qa_chain.invoke(query)
    return response

if __name__ == "__main__":
    qa_chain = load_qa_chain()
    
    while True:
        user_query = input("Enter your question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        response = query_data(qa_chain, user_query)
        print("Response:", response)
        print()