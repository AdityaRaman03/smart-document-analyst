from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")



"""query = input("Enter your query")
result=vectorstore.similarity_search(query, k=3)"""

model = ChatGroq(model="llama-3.3-70b-versatile", max_retries=5)

chat_history = []

def generate_q(query, chat_history):
    
    vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    )
    # Rewrite question to be standalone using chat history
    if chat_history:
        rewrite_messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Return only the rewritten question."),
            *chat_history,
            HumanMessage(content=f"New question: {query}"),
        ]
        search_query = model.invoke(rewrite_messages).content
        print(f"Revised question: {search_query}")

    else:
        search_query = query

    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_query)

    # Build context outside the f-string to avoid backslash issues
    context = "\n".join(f"- {doc.page_content}" for doc in docs)

    combined_input = (
        f"Based on the following documents, please answer this question: {query}\n\n"
        f"Documents:\n{context}\n\n"
        "Please provide a clear, helpful answer using only the information from these documents. "
        'If you can\'t find the answer in the documents, say "I don\'t have enough information '
        'to answer that question based on the provided documents."'
    )

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
        *chat_history,
        HumanMessage(content=combined_input),
    ]

    answer = model.invoke(messages).content

    # Append original question (not the rewritten one) to history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer, docs


def Start_chat():

    print( "Ask Questions. Print Exit to Quit")

    while True:
        query = input("Enter your query")

        if query == "Exit":
            print("Exiting")
            break

        generate_q(query, chat_history)

if __name__ == "__main__":
    Start_chat()