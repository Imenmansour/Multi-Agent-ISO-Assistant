# Import necessary libraries
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import json
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_ollama import ChatOllama



llm_ollama = ChatOllama(model="qwen3:0.6b",temperature=0.7)

model_id=r"C:\Users\G800613RTS\bge-reranker-v2-m3"
reranker_tokenizer = AutoTokenizer.from_pretrained(model_id)
reranker_model = AutoModelForSequenceClassification.from_pretrained(model_id)
reranker_model.eval()
# Reranker function
def rerank_documents(query, documents, top_k=2):
    doc_texts = []
    docs_list = []
    
    for doc in documents:
        # Handle tuple from similarity_search_with_score
        if isinstance(doc, tuple) and len(doc) == 2:
            document = doc[0]  # Extract the document from the (doc, score) tuple
            docs_list.append(document)
            
            if hasattr(document, 'page_content'):
                doc_texts.append(document.page_content)
            elif isinstance(document, dict) and 'text' in document:
                doc_texts.append(document['text'])
            elif isinstance(document, dict) and 'chunk' in document:
                doc_texts.append(document['chunk'])
            elif isinstance(document, dict) and 'summary_chunk' in document:
                doc_texts.append(document['summary_chunk'])
            elif isinstance(document, dict) and 'content' in document:
                doc_texts.append(document['content'])
            elif isinstance(document, str):
                doc_texts.append(document)
            else:
                print(f"Warning: Skipping unsupported document type in tuple: {type(document)}")
                continue
        # Original handling for direct documents
        elif hasattr(doc, 'page_content'):
            doc_texts.append(doc.page_content)
            docs_list.append(doc)
        elif isinstance(doc, dict) and 'text' in doc:
            doc_texts.append(doc['text'])
            docs_list.append(doc)
        elif isinstance(doc, dict) and 'chunk' in doc:
            doc_texts.append(doc['chunk'])
            docs_list.append(doc)
        elif isinstance(doc, dict) and 'summary_chunk' in doc:
            doc_texts.append(doc['summary_chunk'])
            docs_list.append(doc)
        elif isinstance(doc, dict) and 'content' in doc:
            doc_texts.append(doc['content'])
            docs_list.append(doc)
        elif isinstance(doc, str):
            doc_texts.append(doc)
            docs_list.append(doc)
        else:
            print(f"Warning: Skipping unsupported document type: {type(doc)}")
            continue

    pairs = [[query, text] for text in doc_texts]

    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()

    scored_docs = list(zip(scores.tolist(), docs_list))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]
# Load embeddings model
print("Loading embedding model...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)

class CustomSentenceTransformerEmbeddings(Embeddings):  # Inherit from LangChain base
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

embeddings = CustomSentenceTransformerEmbeddings(model)
# Connect to Vector Stores
print("Connecting to vector stores...")

# ISO 27001 Vector Stores
vector_store_i_title = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="io_title_collection",
    url="http://localhost:6333",
)
vector_store_i_text = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="io_text_collection",
    url="http://localhost:6333",
)

# ISO 42001 Vector Stores
vector_store_42001_originals = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="io42001_text_collection",
    url="http://localhost:6333",
)
vector_store_iso_title24001 = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="io42001_title_collection",
    url="http://localhost:6333",
)
#-------------------
#tools
#-------------------
# Define the retrieval tools
@tool(response_format="content_and_artifact")
def retrieve27001(query: str):
    """Retrieve information related to ISO 27001 query."""
    
    # Step 1: Retrieve from vector store
    retrieved_docs = vector_store_i_text.similarity_search(query, k=5)
    print("\n📥 Top 5 Retrieved ISO 27001 Documents (Before Reranking):")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Step 2: Rerank the top results
    reranked_docs = rerank_documents(query, retrieved_docs, top_k=2)
    print("\n🔝 Top 2 Reranked ISO 27001 Documents:")
    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Step 3: Prepare serialized output
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in reranked_docs
    )
    
    return serialized, reranked_docs

@tool(response_format="content_and_artifact")
def retrieve27002(query: str):
    """Retrieve information related to ISO 27002 query."""
    
    # Step 1: Retrieve from vector store - using the same vector store as 27001 for now
    # You can replace with a dedicated 27002 vector store if available
    retrieved_docs = vector_store_i_text.similarity_search(query, k=5)
    print("\n📥 Top 5 Retrieved ISO 27002 Documents (Before Reranking):")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Step 2: Rerank the top results
    reranked_docs = rerank_documents(query, retrieved_docs, top_k=2)
    print("\n🔝 Top 2 Reranked ISO 27002 Documents:")
    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Step 3: Prepare serialized output
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in reranked_docs
    )
    
    return serialized, reranked_docs

@tool(response_format="content_and_artifact")
def retrieve42001(query: str):
    """Retrieve information related to ISO 42001 query."""
 
    # Step 1: Retrieve from ISO 42001 vector store
    retrieved_docs = vector_store_42001_originals.similarity_search(query, k=5)
    print("\n📥 Top 5 Retrieved ISO 42001 Documents (Before Reranking):")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Step 2: Rerank the top results
    reranked_docs = rerank_documents(query, retrieved_docs, top_k=2)
    print("\n🔝 Top 2 Reranked ISO 42001 Documents:")
    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Step 3: Prepare serialized output
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in reranked_docs
    )
    
    return serialized, reranked_docs

# Create a ToolNode with all retrieve tools
# The LLM will decide which tool to use based on the query content
tools = [retrieve27001, retrieve27002, retrieve42001]
tool_node = ToolNode(tools)
# Build the graph
print("Building ISO Q&A graph with dynamic tool selection...")
graph_builder = StateGraph(MessagesState)

def query_or_respond(state: MessagesState):
    """Let the LLM determine which ISO standard tool to use based on the query content."""
    # Get the latest message
    last_message = state["messages"][-1] if state["messages"] else None
    
    if last_message and hasattr(last_message, "content"):
        print(f"[INFO] Processing query: {last_message.content}")
        
        # Bind all tools to the LLM and let it choose which one to use
        # The LLM will select the appropriate retrieval tool based on the query content
        llm_with_tools = llm_ollama.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
        
        
    
        return {"messages": [response]}
    
    # Fallback if no message is found
    return {"messages": []}

def generate(state: MessagesState):
    """Generate answer based on retrieved content."""
    # Get the most recent tool messages
    recent_tool_messages = [m for m in reversed(state["messages"]) if m.type == "tool"]
    tool_messages = recent_tool_messages[::-1]
    
    
    # Format the tool output into a context for the response
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    
    
    system_message_content = (
       
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    
    # Get relevant conversation messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    # Prepare prompt and generate response
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm_ollama.invoke(prompt)
    
    
    
    return {"messages": [response]}
#--------------
#Graph
#--------------
# Build the graph
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate", generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
memory = MemorySaver()

# Compile the graph
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}


# Nothing here - we only need one graph

def ask_iso_question(user_query: str, iso_standard: str = "27002", thread_id: str = "thread-001"):
    """
    Direct RAG implementation for ISO documentation Q&A.
    Retrieves relevant documents, reranks them, and generates a response.
    
    Args:
        user_query: The query to answer
        iso_standard: The ISO standard to use (27001, 27002, or 42001)
        thread_id: The conversation thread ID
        
    Returns:
        A tuple containing the response and sources
    """
    print(f"[INFO] Processing ISO {iso_standard} query: {user_query}")
    
    # Select the appropriate vector stores and prompt based on the ISO standard
    if iso_standard == "27001":
        title_store = vector_store_i_title
        text_store = vector_store_i_text
        standard_name = "ISO 27001"
    elif iso_standard == "42001":
        title_store = vector_store_iso_title24001
        text_store = vector_store_42001_originals
        standard_name = "ISO 42001"
    else:  # Default to 27002
        title_store = vector_store_i_title  # Using same as 27001 for now, replace if dedicated store is available
        text_store = vector_store_i_text  # Using same as 27001 for now, replace if dedicated store is available
        standard_name = "ISO 27002"
    
    # Step 1: Initial retrieval from vector store
    results = title_store.similarity_search(query=user_query, k=2)
    print(f"[DEBUG] Retrieved {len(results)} documents")
    
    # Step 2: Debug each result
    for r in results:
        print(f"[DEBUG] Doc Source: {r.metadata.get('source')}, Title: {r.metadata.get('title')}")
    
    # Step 3: Rerank using reranker
    reranked_docs = rerank_documents(user_query, results, top_k=1)
    
    # Step 4: Extract and format document content for context
    formatted_docs = []
    sources = set()
    
    for doc in reranked_docs:
        source = doc.metadata.get('source', 'Unknown source')
        title = doc.metadata.get('title', 'No title available')
        
        # Add debug prints for the extracted metadata
        print(f"[DEBUG] Processing document - Source: {source}")
        print(f"[DEBUG] Processing document - Title: {title}")
        
        formatted_doc = f"Source: {source}\nTitle: {title}\nContent: {doc.page_content}"
        formatted_docs.append(formatted_doc)
        sources.add(source)
        sources.add(title)
        
        # Print the formatted document string
        print(f"[DEBUG] Formatted document: {formatted_doc[:200]}...")
    
    # Combine all document content for context
    context = "\n\n".join(formatted_docs)
    
    # Step 5: Generate response using LLM with context
    system_prompt = (
        f"You are an assistant for {standard_name} questions. "
        "Use the following context from ISO documents to answer the user's question. "
        "Make sure to include the source of any information you use in your response.\n\n" + context
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Generate response
    response = llm_ollama.invoke(messages).content
    
    print(f"\n🔹 {standard_name} Sources used:", sources)
    
    # Return both the response text and the sources used
    return response, list(sources)