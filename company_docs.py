from typing import List
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
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
from transformers import AutoTokenizer
from langchain_ollama import ChatOllama
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from typing import List
print("Setting up graphs and language models...")

# Define the database file path for chat history
DB_FILE = "db.json"



# -----------------------------
# Initialize Language Models
# -----------------------------
print("Initializing language models...")



llm_ollama = ChatOllama(model="qwen3:0.6b",temperature=0.7)

gemma_model=ChatOllama(model="qwen3:0.6b",temperature=0.7)
model_id=r"C:\Users\G800613RTS\bge-reranker-v2-m3"
reranker_tokenizer = AutoTokenizer.from_pretrained(model_id)
reranker_model = AutoModelForSequenceClassification.from_pretrained(model_id)
reranker_model.eval()
#reranker 
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
# -----------------------------
# Reload embeddings model
# -----------------------------
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

# -----------------------------
# Connect to Vector Stores
# -----------------------------
vector_store_summaries= QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="co_summary_vectors",
    url="http://localhost:6333",
)
vector_store_originals = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="co_original_vectors",
    url="http://localhost:6333",
)
# -----------------------------
# COMP Q&A Tool Setup
# -----------------------------
@tool(response_format="content_and_artifact")
def retrieve_company(query: str):
    """Retrieve and rerank relevant company documents based on a query."""
    # Step 1: Initial retrieval from vector store
    results = vector_store_originals.similarity_search(query, k=10)

    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] Retrieved {len(results)} documents")

    # Step 2: Debug each result
    for r in results:
        print(f"[DEBUG] Doc Source: {r.metadata.get('source')}, Summary: {r.metadata.get('summary')[:100]}")

    # Step 3: Rerank using your reranker
    reranked_docs = rerank_documents(query, results, top_k=5)

    # Step 4: Format output
    formatted = "\n\n".join(
        f"Source: {doc.metadata.get('source')}\n"
        f"Summary: {doc.metadata.get('summary')}\n"
        f"Original: {doc.page_content}"
        for doc in reranked_docs
    )

    return formatted, reranked_docs


# -----------------------------
# COMP Q&A Graph
# -----------------------------
print("Building Company Q&A graph...")
graph_builder_company = StateGraph(MessagesState)

# Node: Call LLM with tools


# Node: Final response based on tool output
def ask_company_question(user_query: str, thread_id: str = "thread-002"):
    """
    Direct RAG implementation for company documentation Q&A.
    Retrieves relevant documents, reranks them, and generates a response.
    """
    print(f"[INFO] Processing company query: {user_query}")
    
    # Step 1: Initial retrieval from vector store
    results = vector_store_originals.similarity_search(query=user_query, k=15)
    print(f"[DEBUG] Retrieved {len(results)} documents")
    
    # Step 2: Debug each result
    for r in results:
        print(f"[DEBUG] Doc Source: {r.metadata.get('source')}, Summary: {r.metadata.get('summary')[:100]}")
    
    # Step 3: Rerank using reranker
    reranked_docs = rerank_documents(user_query, results, top_k=10)
    
    # Step 4: Extract and format document content for context
    formatted_docs = []
    sources = set()
    
    for doc in reranked_docs:
        source = doc.metadata.get('source', 'Unknown source')
        summary = doc.metadata.get('summary', 'No summary available')
        title = doc.metadata.get('title', '')
        
        formatted_docs.append(f"Source: {source}\nSummary: {summary}\nOriginal: {doc.page_content}")
        
        # Add both source and title to the sources
        sources.add(source)
        
       
    
    # Combine all document content for context
    context = "\n\n".join(formatted_docs)
    
    # Step 5: Generate response using LLM with context
    system_prompt = (
    "You are a responsable de gestion de l'information (information management officer) tasked with answering questions. "
    "IMPORTANT: Answer ONLY based on the context from company documents provided below. "
    "Do NOT use any external knowledge or make assumptions beyond what's explicitly stated in the context. "
    "If the answer to the user's question is not found in the provided context, respond with: "
    "'Je ne trouve pas d'information sur cette question dans les documents disponibles.' "
    "Make sure to include the source of any information you use in your response.\n\n" + context
)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Generate response
    response = llm_ollama.invoke(messages).content
    
    print("\n🔹 Sources used:", sources)

    return response, list(sources)



# Create tool node directly
tools_company = ToolNode([retrieve_company])
graph_builder_company.add_node("tools", tools_company)
graph_builder_company.add_node("generate", ask_company_question)

# Set entry point directly to the tools node
graph_builder_company.set_entry_point("tools")

# Connect tools node to generate node
graph_builder_company.add_edge("tools", "generate")
graph_builder_company.add_edge("generate", END)

# Attach memory for threaded sessions
company_memory = MemorySaver()
company_graph = graph_builder_company.compile(checkpointer=company_memory)

