import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langgraph.graph import StateGraph, END

def setup_agent():
    # 1. Load API Keys
    load_dotenv()

    # 2. Define our AI Model and Embedding Model
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Load your REAL Document (Replacing the video's text strings)
    print("Loading PDF...")
    loader = PDFPlumberLoader("sample_document.pdf") # Ensure this file is in your folder
    raw_docs = loader.load()

    # 4. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = text_splitter.split_documents(raw_docs)

    # 5. Store in a Vector Database
    print("Creating Vector Database...")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    #----------------------------------------------------------------------------------------------------------------------------
    # Define the Agent's "Brain" (The State)
    # LangGraph relies on a "State" dictionary that gets passed from agent to agent. It acts as the memory for the pipeline

    class AgentState(TypedDict):
        question: str
        chat_history: list
        rewritten_question: str
        documents: list
        generation: str
        relevance: str
        retry_count: int

    # Node 1: Rewrite the Query
    def rewrite_query(state: AgentState):
        history = "\n".join(state.get("chat_history", []))
        prompt = ChatPromptTemplate.from_template(
            "You are a query optimization agent. Look at the Chat History and the new User Question. "
            "Rewrite the User Question into a standalone, specific query for a database search. "
            "If the question uses pronouns like 'it' or 'they', replace them with the actual subject from the history.\n\n"
            "Chat History:\n{history}\n\n"
            "User Question: {question}\n\n"
            "Rewritten Query:"
        )
        chain = prompt | llm
        result = chain.invoke({"history": history, "question": state["question"]})
    
        # We print it so you can see the AI's internal thoughts!
        print(f"   [Internal Thought] Rewrote query to: {result.content.strip()}")
        return {"rewritten_question": result.content}

    # Node 2: Retrieve Documents
    def retrieve(state: AgentState):
        query = state.get("rewritten_question", state["question"])
        docs = retriever.invoke(query)
        return {"documents": docs}

    # Node 3: Grade the Documents (Check if they are actually useful)
    def grade_documents(state: AgentState):
        context = "\n".join([d.page_content for d in state["documents"]])
        prompt = ChatPromptTemplate.from_template(
            "You are a strict relevance evaluator. Determine if the context is sufficient to answer the question. "
            "Respond ONLY with 'relevant' or 'irrelevant'.\nContext: {context}\nQuestion: {question}"
        )
        chain = prompt | llm
        result = chain.invoke({"context": context, "question": state["question"]})
    
        # We use .lower() to fix the bug from the video!
        grade = result.content.strip().lower()
        return {"relevance": "relevant" if "relevant" in grade else "irrelevant"}

    # Node 4: Generate the Final Answer
    def generate(state: AgentState):
        context = "\n".join([d.page_content for d in state["documents"]])
        prompt = ChatPromptTemplate.from_template(
            "Answer the question using ONLY the provided context.\nContext: {context}\nQuestion: {question}"
        )
        chain = prompt | llm
        result = chain.invoke({"context": context, "question": state["question"]})
        return {"generation": result.content}

    # Node 5: Track Retries to prevent infinite loops
    def retry_tracker(state: AgentState):
        count = state.get("retry_count", 0)
        return {"retry_count": count + 1}
    #----------------------------------------------------------------------------------------------------------------------------

    # Wire it all together with LangGraph
    # This is where the magic happens. We connect the nodes using "edges" and "conditional edges" to create the reasoning loop.

    # Define the workflow graph
    workflow = StateGraph(AgentState)

    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("retry", retry_tracker)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")

    # The Conditional Logic: Decide what to do based on the Grade
    def decide_to_generate_or_retry(state: AgentState):
        if state["relevance"] == "relevant" or state.get("retry_count", 0) >= 2:
            return "generate"
        return "retry"

    workflow.add_conditional_edges(
        "grade", 
        decide_to_generate_or_retry,
        {
            "generate": "generate", # If function returns "generate", go to "generate" node
            "retry": "retry"        # If function returns "retry", go to "retry" node
        }
    )
    workflow.add_edge("retry", "rewrite") # If it fails, loop back to rewrite!
    workflow.add_edge("generate", END)

    # Compile the Graph
    app = workflow.compile()
    return app


if __name__ == "__main__":
    app = setup_agent()
    print("\n" + "="*50)
    print("🧠 Agentic RAG Chatbot (Memory + Table Parsing)")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    chat_memory = []
    
    # The while loop keeps the program running continuously
    while True:
        # 1. Get the user's question
        user_question = input("\nYou: ")
        
        # 2. Check if the user wants to exit
        if user_question.lower() in ['exit', 'quit']:
            print("Goodbye! Shutting down...")
            break
            
        # 3. Ignore empty questions if you accidentally hit Enter
        if not user_question.strip():
            continue
        
        # 4. Feed the question to the LangGraph agent
        inputs = {
            "question": user_question, 
            "chat_history": chat_memory,
            "retry_count": 0
        }
        
        print("Agent is searching and thinking...")
        result = app.invoke(inputs)
        answer = result["generation"]
        
        print(f"\nAgent: {answer}")
        
        # NEW: After answering, save the interaction to memory
        chat_memory.append(f"User: {user_question}")
        chat_memory.append(f"Agent: {answer}")
        
        # Keep memory from getting too long (keeps the last 4 interactions)
        if len(chat_memory) > 8:
            chat_memory = chat_memory[-8:]