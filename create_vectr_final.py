import os
import json
from uuid import uuid4
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# --- Embedding Model Setup ---
print("Loading embedding model...")
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)

class CustomSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

embeddings = CustomSentenceTransformerEmbeddings(model)

# --- Qdrant Setup ---
collection_name = "android-collection"
qdrant_url = "http://localhost:6333"
client = QdrantClient(url=qdrant_url)

# --- Data Folders ---
folders = [
    r"C:\Users\G800613RTS\Desktop\Android_Data\API31_DATA",
    r"C:\Users\G800613RTS\Desktop\Android_Data\API32_DATA",
    r"C:\Users\G800613RTS\Desktop\Android_Data\API34_DATA",
    r"C:\Users\G800613RTS\Desktop\Android_Data\Common_Data"
]

# --- Check collection exists ---
existing = [c.name for c in client.get_collections().collections]
if collection_name not in existing:
    raise Exception(f"Collection '{collection_name}' does not exist. Please create it first.")

def process_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return []
    if not isinstance(data, list):
        print(f"❌ File {filepath} does not contain a list.")
        return []
    return data

def embed_and_upsert(item):
    text_to_embed = item.get("text_format", "")
    markdown_data = item.get("markdown_format", "")
    url = item.get("url", "")

    if not text_to_embed.strip():
        print("⚠️ Skipping item with empty text_format.")
        return

    # Generate embedding using your SentenceTransformer
    embedding = embeddings.embed_query(text_to_embed)

    # Prepare payload: store markdown and url as metadata
    payload = {
        "markdown_format": markdown_data,
        "url": url,
        "text_format": text_to_embed  # Optionally keep for reference
    }

    # Upsert to Qdrant
    point_id = str(uuid4())
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        print(f"✅ Upserted: {point_id}")
    except Exception as e:
        print(f"❌ Qdrant upsert error: {e}")

def process_folder(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".json"):
                filepath = os.path.join(root, file)
                print(f"📄 Processing {filepath}")
                items = process_json_file(filepath)
                for item in items:
                    embed_and_upsert(item)

if __name__ == "__main__":
    for folder in folders:
        process_folder(folder)
    print("🎉 All data processed and upserted to Qdrant!")
# import sys
# import streamlit as st
# import os
# import json
# from pathlib import Path
# from typing import List, TypedDict, Literal
# from langchain_core.messages import SystemMessage
# from langchain_core.documents import Document
# from langchain_core.tools import tool
# from langgraph.graph import END, START, StateGraph
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.graph import MessagesState
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from langchain.prompts import ChatPromptTemplate
# from operator import itemgetter
# from sentence_transformers import SentenceTransformer
# from langchain.embeddings.base import Embeddings
# from langchain.storage import InMemoryByteStore
# from qdrant_client.http import models as rest
# from qdrant_client.http.models import Distance, VectorParams
# import uuid
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain.storage import InMemoryByteStore
# from langchain_core.documents import Document
# import json
# from qdrant_client import QdrantClient, models
# from uuid import uuid4
# import re
# import pandas as pd
# from qdrant_client.http.models import CountRequest
# import os
# from docx import Document as DocxDocument
# from docx.shared import RGBColor
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from langchain_ollama import ChatOllama
# from transformers import AutoModelForCausalLM, AutoTokenizer
# print("Setting up graphs and language models...")

# Define the database file path for chat history
# DB_FILE = "db.json"



# -----------------------------
# Initialize Language Models
# -----------------------------
# print("Initializing language models...")



# -----------------------------
# Reload embeddings model
# -----------------------------
# print("Loading embedding model...")
# model = SentenceTransformer(
#     "nomic-ai/nomic-embed-text-v1",
#     trust_remote_code=True
# )

# class CustomSentenceTransformerEmbeddings(Embeddings):  # Inherit from LangChain base
#     def __init__(self, model):
#         self.model = model

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, normalize_embeddings=True).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode([text], normalize_embeddings=True)[0].tolist()

# embeddings = CustomSentenceTransformerEmbeddings(model)
# def create_collection_if_not_exists(client, collection_name, vector_size):
#     existing = [c.name for c in client.get_collections().collections]
#     if collection_name not in existing:
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config=models.VectorParams(
#                 size=vector_size,
#                 distance=models.Distance.COSINE
#             )
#         )
#         print(f"✅ Created collection: {collection_name}")
#     else:
#         print(f"✅ Collection already exists: {collection_name}")
# -----------------------------
# Connect to Vector Stores
# -----------------------------
# client_iso = QdrantClient(url="http://localhost:6333")
# create_collection_if_not_exists(client_iso, "io42001_title_collection", 768)

# with open("iso_42001.json", "r",encoding="utf-8") as f:
#     iso_data = json.load(f)

# title_documents = [
#     Document(
#         page_content=chunk["title"],  # Only embed the title
#         metadata={
#             "title": chunk["title"],
#             "content": chunk["text"],  # Store full text as metadata
#             "source": chunk["source"]
#         }
#     )
#     for chunk in iso_data
# ]

# vector_store_iso_title = QdrantVectorStore(
#     client=client_iso,
#     collection_name="io42001_title_collection",
#     embedding=embeddings,
#     validate_embeddings=False
# )

# vector_store_iso_title.add_documents(documents=title_documents, ids=[str(uuid4()) for _ in title_documents])


# -----------------------------
# ISO Vector Store (Full Text Embedding)
# -----------------------------
# create_collection_if_not_exists(client_iso, "io42001_text_collection", 768)

# full_text_documents = [
#     Document(
#         page_content=chunk["text"],
#         metadata={
#             "title": chunk["title"],
#             "source": chunk["source"]
#         }
#     )
#     for chunk in iso_data
# ]

# vector_store_iso = QdrantVectorStore(
#     client=client_iso,
#     collection_name="io42001_text_collection",
#     embedding=embeddings,
#     validate_embeddings=False
# )

# vector_store_iso.add_documents(documents=full_text_documents, ids=[str(uuid4()) for _ in full_text_documents])


# # -----------------------------
# # Company Documents (Summary + Original Embeddings)
# # -----------------------------
# # Define folders
# summary_folder = Path(r"C:\Users\g545453\Desktop\Chatbot\Company_json")
# original_folder = Path(r"C:\Users\g545453\Desktop\Chatbot\Company_json_1")

# # Load all JSON data from both folders
# def load_json_files_from_folder(folder_path):
#     chunks = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".json"):
#             with open(folder_path / file_name, "r", encoding="utf-8") as f:
#                 try:
#                     data = json.load(f)
#                     if isinstance(data, list):
#                         chunks.extend(data)
#                     else:
#                         chunks.append(data)
#                 except json.JSONDecodeError:
#                     print(f"Skipping invalid JSON: {file_name}")
#     return chunks

# # Load summary and original chunks
# summary_chunks = load_json_files_from_folder(summary_folder)
# original_chunks = load_json_files_from_folder(original_folder)



# # Create summary documents
# summary_docs = [
#     Document(
#         page_content=chunk["summary"],
#         metadata={
#             "chunk_text": chunk["content"],
#             "source": chunk["name_pdf"]
#         }
#     )
#     for chunk in summary_chunks
# ]

# # Create original documents
# original_docs = [
#     Document(
#         page_content=chunk["content"],
#         metadata={
#             "summary": chunk["summary"],
#             "source": chunk["name_pdf"]
#         }
#     )
#     for chunk in original_chunks
# ]

# # Create collections if they don't exist
# collection_summary = "co_summary_vectors"
# collection_original = "co_original_vectors"
# create_collection_if_not_exists(client_iso, collection_summary, 768)
# create_collection_if_not_exists(client_iso, collection_original, 768)

# # Initialize vector stores
# vector_store_summaries = QdrantVectorStore(
#     client=client_iso,
#     collection_name=collection_summary,
#     embedding=embeddings,
#     validate_embeddings=False
# )

# vector_store_originals = QdrantVectorStore(
#     client=client_iso,
#     collection_name=collection_original,
#     embedding=embeddings,
#     validate_embeddings=False
# )
# summary_ids = [str(uuid.uuid4()) for _ in range(len(summary_docs))]
# original_ids = [str(uuid.uuid4()) for _ in range(len(original_docs))]
# # Upload to vector stores
# vector_store_summaries.add_documents(documents=summary_docs, ids=summary_ids)
# vector_store_originals.add_documents(documents=original_docs, ids=original_ids)
# # -----------------------------
# # ISO Q&A Tool Setup
# # -----------------------------
# @tool(response_format="content_and_artifact")
# def retrieve(query: str):
#     """Retrieve information related to a query."""
    
#     # Step 1: Retrieve from vector store
#     retrieved_docs = vector_store_iso.similarity_search(query, k=5)
#     print("\n📥 Top 5 Retrieved Documents (Before Reranking):")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Doc {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
#         print(f"Content Preview: {doc.page_content[:200]}...\n")

#     # Step 2: Rerank the top results
#     reranked_docs = rerank_documents(query, retrieved_docs, top_k=2)
#     print("\n🔝 Top 2 Reranked ISO Documents:")
#     for i, doc in enumerate(reranked_docs):
#         print(f"Rank {i+1}: Source: {doc.metadata.get('source')}, Title: {doc.metadata.get('title')}")
#         print(f"Content Preview: {doc.page_content[:200]}...\n")

#     # Step 3: Prepare serialized output
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in reranked_docs
#     )
    
#     return serialized, reranked_docs


# # -----------------------------
# # ISO Q&A Graph
# # -----------------------------
# print("Building ISO Q&A graph...")
# graph_builder = StateGraph(MessagesState)

# def query_or_respond(state: MessagesState):
#     llm_with_tools = gemma_model.bind_tools([retrieve])
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": [response]}

# def generate(state: MessagesState):
#     recent_tool_messages = [m for m in reversed(state["messages"]) if m.type == "tool"]
#     docs_content = "\n\n".join(doc.content for doc in recent_tool_messages[::-1])
#     prompt = [SystemMessage(
#         "You are an assistant for ISO 27002 questions. "
#         "Use this context:\n\n" + docs_content
#     )] + [m for m in state["messages"] if m.type in ("human", "system")]
    
#     response = gemma_model.invoke(prompt)
#     return {"messages": [response]}

# tools = ToolNode([retrieve])
# graph_builder.add_node("query_or_respond", query_or_respond)
# graph_builder.add_node("tools", tools)
# graph_builder.add_node("generate", generate)
# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges("query_or_respond", tools_condition, {
#     END: END,
#     "tools": "tools"
# })
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)
# memory = MemorySaver()
# iso_graph = graph_builder.compile(checkpointer=memory)

# def ask_iso_question(user_query: str, thread_id: str = "thread-001"):
#     config = {"configurable": {"thread_id": thread_id}}
#     final_response = None
#     for step in iso_graph.stream(
#         {"messages": [{"role": "user", "content": user_query}]},
#         stream_mode="values",
#         config=config
#     ):
#         final_response = step["messages"][-1]
#     return final_response.content if final_response else "No response generated."



# # -----------------------------
# # COMP Q&A Tool Setup
# # -----------------------------
# @tool(response_format="content_and_artifact")
# def retrieve_company(query: str):
#     """Retrieve and rerank relevant company documents based on a query."""
#     # Step 1: Initial retrieval from vector store
#     results = vector_store_originals.similarity_search(query, k=10)

#     print(f"[DEBUG] Query: {query}")
#     print(f"[DEBUG] Retrieved {len(results)} documents")

#     # Step 2: Debug each result
#     for r in results:
#         print(f"[DEBUG] Doc Source: {r.metadata.get('source')}, Summary: {r.metadata.get('summary')[:100]}")

#     # Step 3: Rerank using your reranker
#     reranked_docs = rerank_documents(query, results, top_k=5)

#     # Step 4: Format output
#     formatted = "\n\n".join(
#         f"Source: {doc.metadata.get('source')}\n"
#         f"Summary: {doc.metadata.get('summary')}\n"
#         f"Original: {doc.page_content}"
#         for doc in reranked_docs
#     )

#     return formatted, reranked_docs


# # -----------------------------
# # COMP Q&A Graph
# # -----------------------------
# print("Building Company Q&A graph...")
# graph_builder_company = StateGraph(MessagesState)

# # Node: Call LLM with tools
# def query_or_respond_company(state: MessagesState):
#     llm_with_tools = gemma_model.bind_tools([retrieve_company])
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": [response]}

# # Node: Final response based on tool output
# def generate_company_response(state: MessagesState):
#     # Retrieve the tool responses (which include the formatted document context)
#     recent_tool_messages = [m for m in reversed(state["messages"]) if m.type == "tool"]

#     # Extract the formatted output (summary + content + source) from the tool messages
#     docs_content = "\n\n".join(doc.content for doc in recent_tool_messages[::-1])

#     # 🖨️ Print the document content that will be passed to the LLM
#     print("\n🧾 Retrieved documents used to generate final response:\n")
#     print(docs_content)
#     print("\n--------------------------------------------\n")

#     # Build the prompt with the context, including summaries, content, and sources
#     prompt = [SystemMessage(
#         "You are an assistant for question-answering tasks. "
#         "Use the following context from company documents to answer. If you don't know the answer, say that you don't know in three sentences maximum and keep the answer concise:\n\n" + docs_content
#     )] + [m for m in state["messages"] if m.type in ("human", "system")]

#     # Get the response from the LLM based on the context
#     response = gemma_model.invoke(prompt)

#     # Retrieve the source information from the documents used
#     sources = "\n".join(
#         f"Source: {doc.metadata.get('source')}"
#         for doc in recent_tool_messages
#     )

#     # Append the sources to the generated response
#     full_response = f"{response.content}\n\n{sources}"

#     # Return the final response with the sources
#     return {"messages": [{"content": full_response}]}


# tools_company = ToolNode([retrieve_company])
# graph_builder_company.add_node("query_or_respond", query_or_respond_company)
# graph_builder_company.add_node("tools", tools_company)
# graph_builder_company.add_node("generate", generate_company_response)

# graph_builder_company.set_entry_point("query_or_respond")
# graph_builder_company.add_conditional_edges("query_or_respond", tools_condition, {
#     END: END,
#     "tools": "tools"
# })
# graph_builder_company.add_edge("tools", "generate")
# graph_builder_company.add_edge("generate", END)

# # Attach memory for threaded sessions
# company_memory = MemorySaver()
# company_graph = graph_builder_company.compile(checkpointer=company_memory)

# def ask_company_question(user_query: str, thread_id: str = "thread-002"):
#     config = {"configurable": {"thread_id": thread_id}}
#     final_response = None
#     for step in company_graph.stream(
#         {"messages": [{"role": "user", "content": user_query}]},
#         stream_mode="values",
#         config=config
#     ):
#         final_response = step["messages"][-1]
#     return final_response.content if final_response else "No response generated."


# #Graph

# print("Setting up routing logic...")
# class RouteQuery(TypedDict):
#     destination: Literal["iso_qna", "gap_analysis", "company_docs", "quiz_generation"]

# route_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a routing assistant.
# Determine if the user is asking about:
# 1. ISO standard information (iso_qna)
# 2. A gap analysis request (gap_analysis)
# 3. Company-specific document information (company_docs)
# 4. Quiz generation request (quiz_generation)

# Consider the entire conversation history when making your decision.

# Respond strictly in JSON format as follows:
# {{"destination": "iso_qna"}} or {{"destination": "gap_analysis"}} or {{"destination": "company_docs"}} or {{"destination": "quiz_generation"}}"""),
#     ("human", "{query}"),
#     ("human", "Previous conversation: {history}")
# ])

# route_chain = (
#     route_prompt
#     | gemma_model.with_structured_output(RouteQuery, include_raw=False)
#     | itemgetter("destination")
# )

# # -----------------------------
# # Gap Analysis Logic
# # -----------------------------
# print("Setting up gap analysis logic...")
# # Load theme clauses
# with open("sheets_dict.json", "r") as f:
#     theme_clauses = json.load(f)

# import re
# def parse_gap_analysis_result(result_text):
#     try:
#         # Extract alignment status
#         alignment_match = re.search(r"(?:Alignment Status:|Status:).*?(confirmed|not confirmed)",
#                                    result_text, re.IGNORECASE)
#         alignment_status = alignment_match.group(1).strip() if alignment_match else "Unknown"

#         # Extract gap analysis
#         gap_match = re.search(r"(?:Gap:|Gap Analysis:)(.+?)(?:Recommendation:|References:|$)",
#                              result_text, re.DOTALL)
#         gap_analysis = gap_match.group(1).strip() if gap_match else "No gap analysis provided"

#         # Extract recommendations
#         rec_match = re.search(r"(?:Recommendation:|Recommendations:)(.+?)(?:References:|Company Source:|ISO Source:|$)",
#                              result_text, re.DOTALL)
#         recommendations = rec_match.group(1).strip() if rec_match else "No recommendations provided"

#         return {
#             "alignment_status": alignment_status,
#             "gap_analysis": gap_analysis,
#             "recommendations": recommendations
#         }
#     except Exception as e:
#         print(f"Error parsing gap analysis result: {e}")
#         return {
#             "alignment_status": "Error",
#             "gap_analysis": "Error parsing result",
#             "recommendations": "Error parsing result"
#         }


# def single_document_analysis_prompt(clause, iso_metadata, company_document):
#     iso_titles = iso_metadata.get("titles", ["ISO 27002:2022 Clause"])
#     iso_sources = iso_metadata.get("sources", ["ISO 27002:2022"])
#     iso_contents = iso_metadata.get("contents", ["No content available"])

#     iso_info_combined = "\n".join(
#         f"- Title: {title}\n  Source: {source}\n  Content: {content}"
#         for title, source, content in zip(iso_titles, iso_sources, iso_contents)
#     )

#     prompt = f"""
# You are an assistant tasked with analyzing a single company document in relation to ISO 27002:2022 compliance. 
# Analyze how this specific document addresses the following ISO clause:

# ISO Clause: {clause}

# **ISO 27002:2022 Content:**
# {iso_info_combined}

# **Company Document:**
# Source: {company_document.get('source', 'Unknown')}
# Summary: {company_document.get('summary', 'No summary available')}
# Original Text: {company_document.get('original_text', 'No original text available')}

# Your task is to:
# 1. Identify which key points of the ISO clause this document addresses
# 2. Identify what elements of the ISO clause are missing or not adequately addressed in this document
# 3. Provide recommendations for improving this specific document to better align with the ISO clause

# Do NOT determine overall compliance status. Focus only on this specific document's coverage.

# **Expected Response Format:**
# - **Key Points Covered:** Identify which specific requirements or elements of the ISO clause are addressed in this document.
# - **Missing Elements:** Identify which specific requirements or elements of the ISO clause are missing or inadequately addressed.
# - **Recommendations:** Provide specific recommendations to improve this document to better align with the ISO clause.
# """
#     return prompt

# def aggregation_prompt(clause, iso_metadata, document_analyses):
#     iso_titles = iso_metadata.get("titles", ["ISO 27002:2022 Clause"])
#     iso_sources = iso_metadata.get("sources", ["ISO 27002:2022"])
#     iso_contents = iso_metadata.get("contents", ["No content available"])

#     iso_info_combined = "\n".join(
#         f"- Title: {title}\n  Source: {source}\n  Content: {content}"
#         for title, source, content in zip(iso_titles, iso_sources, iso_contents)
#     )
    
#     # Format the document analyses into a readable format
#     analyses_text = ""
#     for i, analysis in enumerate(document_analyses, 1):
#         analyses_text += f"\n--- Document {i} ---\n"
#         analyses_text += f"Source: {analysis['source']}\n"
#         analyses_text += f"Key Points Covered: {analysis['key_points_covered']}\n"
#         analyses_text += f"Missing Elements: {analysis['missing_elements']}\n"
#         analyses_text += f"Recommendations: {analysis['recommendations']}\n"

#     prompt = f"""
# You are an assistant tasked with aggregating multiple document analyses to determine overall ISO 27002:2022 compliance.
# Review the following individual document analyses related to this ISO clause:

# ISO Clause: {clause}

# **ISO 27002:2022 Content:**
# {iso_info_combined}

# **Individual Document Analyses:**
# {analyses_text}

# Your task is to:
# 1. Review all document analyses collectively
# 2. Determine if the company's documentation as a whole adequately addresses the ISO clause requirements
# 3. Provide an overall compliance assessment and gap analysis
# 4. Offer comprehensive recommendations

# **Expected Response Format:**
# - **Alignment Status:** State whether the company documentation collectively aligns with the ISO clause (confirmed or not confirmed)
# - **Gap Analysis:** Provide a comprehensive analysis of gaps across all documents
# - **Recommendations:** Offer consolidated recommendations to achieve full compliance
# - **References:** List the key company documents that support your conclusion
# """
#     return prompt   
# def save_to_csv(results, filename="gap_analysis_results.csv"):
#     import pandas as pd
#     df = pd.DataFrame(results)
    
#     # Save to the current directory with absolute path
#     import os
#     csv_path = os.path.abspath(filename)
#     df.to_csv(csv_path, index=False)
    
#     print(f"CSV saved to: {csv_path}")
#     return csv_path


# def save_to_docx(results, filename="gap_analysis_results.docx"):
#     doc = DocxDocument()
#     doc.add_heading('ISO 27002:2022 Gap Analysis Report', 0)

#     # Add introduction
#     doc.add_paragraph(
#         "This document presents the results of a gap analysis comparing the organization's documentation "
#         "against the ISO 27002:2022 standard requirements."
#     )

#     # Summary statistics
#     total_clauses = len(results)
#     confirmed_clauses = len([r for r in results if r["alignment_status"].lower() == "confirmed"])
#     not_confirmed_clauses = total_clauses - confirmed_clauses
#     compliance_percentage = (confirmed_clauses / total_clauses) * 100 if total_clauses > 0 else 0

#     doc.add_heading('Executive Summary', level=1)
#     summary_para = doc.add_paragraph()
#     summary_para.add_run(f"Total clauses analyzed: {total_clauses}\n").bold = True
#     summary_para.add_run(f"Confirmed clauses: {confirmed_clauses}\n").bold = True
#     summary_para.add_run(f"Non-compliant clauses: {not_confirmed_clauses}\n").bold = True
#     summary_para.add_run(f"Overall compliance: {compliance_percentage:.1f}%").bold = True

#     # Group results by theme
#     themes = {}
#     for result in results:
#         themes.setdefault(result["theme"], []).append(result)

#     # Detailed results by theme
#     for theme, theme_results in themes.items():
#         doc.add_heading(f'Theme: {theme}', level=1)
#         for result in theme_results:
#             doc.add_heading(result["clause"], level=2)

#             # Alignment status
#             status_para = doc.add_paragraph()
#             status_para.add_run("Alignment Status: ").bold = True
#             status_run = status_para.add_run(result["alignment_status"])
#             if result["alignment_status"].lower() == "confirmed":
#                 status_run.font.color.rgb = RGBColor(0, 128, 0)  # Green
#             else:
#                 status_run.font.color.rgb = RGBColor(255, 0, 0)  # Red

#             # Gap analysis
#             gap_para = doc.add_paragraph()
#             gap_para.add_run("Gap Analysis: ").bold = True
#             gap_para.add_run(result["gap_analysis"])

#             # Recommendations
#             rec_para = doc.add_paragraph()
#             rec_para.add_run("Recommendations: ").bold = True
#             rec_para.add_run(result["recommendations"])

#             # Sources
#             sources_para = doc.add_paragraph()
#             sources_para.add_run("Company Source: ").bold = True
#             sources_para.add_run(result["company_source"])
#             sources_para.add_run("\nISO Source: ").bold = True
#             sources_para.add_run(result["iso_source"])

#             doc.add_paragraph("---")

#     # Convert to absolute path and save
#     abs_path = os.path.abspath(filename)
#     doc.save(abs_path)
#     print(f"📄 DOCX saved to: {abs_path}")
#     return abs_path
# def parse_document_analysis_result(result_text):
#     try:
#         # Extract key points covered
#         key_points_match = re.search(r"(?:Key Points Covered:|Coverage:)(.+?)(?:Missing Elements:|Missing Points:|$)",
#                                    result_text, re.DOTALL)
#         key_points = key_points_match.group(1).strip() if key_points_match else "No key points identified"

#         # Extract missing elements
#         missing_match = re.search(r"(?:Missing Elements:|Missing Points:)(.+?)(?:Recommendations:|$)",
#                              result_text, re.DOTALL)
#         missing_elements = missing_match.group(1).strip() if missing_match else "No missing elements identified"

#         # Extract recommendations
#         rec_match = re.search(r"(?:Recommendation:|Recommendations:)(.+?)(?:$)",
#                              result_text, re.DOTALL)
#         recommendations = rec_match.group(1).strip() if rec_match else "No recommendations provided"

#         return {
#             "key_points_covered": key_points,
#             "missing_elements": missing_elements,
#             "recommendations": recommendations
#         }
#     except Exception as e:
#         print(f"Error parsing document analysis result: {e}")
#         return {
#             "key_points_covered": "Error parsing result",
#             "missing_elements": "Error parsing result",
#             "recommendations": "Error parsing result"
#         }
# print("Precomputing ISO metadata...")
# # Precompute ISO metadata for faster access
# def run_gap_analysis():
#     results = []
#     for theme, clauses in theme_clauses.items():
#         for clause in clauses:
            
#             # Get ISO information
#             iso_docs = vector_store_iso_title.similarity_search_with_score(clause, k=2)
#             iso_chunks = [doc_score[0] for doc_score in iso_docs] if iso_docs else []
            
#             # Combine ISO content, titles, and sources
#             iso_titles = [doc.page_content for doc in iso_chunks]
#             iso_contents = [doc.metadata.get("content", "") for doc in iso_chunks]
#             iso_sources = [doc.metadata.get("source", "") for doc in iso_chunks]
            
#             iso_metadata = {
#                 "titles": iso_titles,
#                 "sources": iso_sources,
#                 "contents": iso_contents
#             }
#              # Print the ISO metadata
#             print("\nISO Metadata Retrieved:")
#             print("Titles:", iso_metadata["titles"])
#             print("Sources:", iso_metadata["sources"])
#             print("Contents:", iso_metadata["contents"])
            
#             # Retrieve and rerank company documents
#             company_chunks = vector_store_summaries.similarity_search_with_score(clause, k=10)
#             reranked_docs = rerank_documents(clause, company_chunks, top_k=7)
            
#             # Store individual document analyses
#             document_analyses = []
            
#             # Analyze each document individually
#             for doc, score in reranked_docs:
#                 company_document = {
#                     'source': doc.metadata.get('source', ''),
#                     'summary': doc.page_content,
#                     'original_text': doc.metadata.get('chunk_text', '')
#                 }
#                 # Print the company document info
                
#                 print("Source:", company_document["source"])
#                 print("Summary:", company_document["summary"])
#                 print("Original Text:", company_document["original_text"])

#                 # Generate and run the prompt for individual document analysis
#                 single_doc_prompt = single_document_analysis_prompt(clause, iso_metadata, company_document)
#                 single_doc_response = gemma_model.invoke(single_doc_prompt).content
                
#                 # Parse the response
#                 parsed_doc_analysis = parse_document_analysis_result(single_doc_response)
                
#                 # Store the analysis with source information
#                 document_analyses.append({
#                     'source': company_document['source'],
#                     'key_points_covered': parsed_doc_analysis['key_points_covered'],
#                     'missing_elements': parsed_doc_analysis['missing_elements'],
#                     'recommendations': parsed_doc_analysis['recommendations']
#                 })
            
#             # Now run the aggregation analysis
#             agg_prompt = aggregation_prompt(clause, iso_metadata, document_analyses)
#             aggregation_response = gemma_model.invoke(agg_prompt).content
            
#             # Parse the aggregation response
#             parsed_aggregation = parse_gap_analysis_result(aggregation_response)
            
#             # Collect sources for reference
#             sources_company = [analysis['source'] for analysis in document_analyses]
            
#             # Collect the final result for the current clause
#             results.append({
#                 "theme": theme,
#                 "clause": clause,
#                 "alignment_status": parsed_aggregation["alignment_status"],
#                 "gap_analysis": parsed_aggregation["gap_analysis"],
#                 "recommendations": parsed_aggregation["recommendations"],
#                 "company_source": sources_company,
#                 "iso_source": iso_titles,
#                 "individual_analyses": document_analyses  # Store individual analyses for reference
#             })
    
#     # Save results
#     csv_path = save_to_csv(results)
#     docx_path = save_to_docx(results)
    
#     import pandas as pd
#     if "gap_analysis_df" not in st.session_state:
#         st.session_state.gap_analysis_df = pd.DataFrame(results)

#     return {"csv": csv_path, "docx": docx_path, "df": results}
# # -----------------------------
# # Quiz generator
# # -----------------------------
# def generate_quiz_question(section, subsection, text):
#     """ 
#     Use LLM to generate a variable number of quiz questions for a specific ISO clause 
#     """
#     # Get the reference ID from the subsection (first part before space) 
#     ref_parts = subsection.split(" ", 1) 
#     ref = ref_parts[0] if len(ref_parts) > 0 else "unknown"
    
#     # Create a more focused prompt for the LLM 
#     prompt = f""" 
# Agissez en tant qu’auditeur principal certifié lead auditor ISO 27002:2022.

# Vous êtes en train d’examiner la clause ISO {subsection} : « {text} ».

# Votre mission est de générer une série réaliste de questions de type audit, suivant une progression logique et en chaîne. Ces questions doivent simuler la manière dont un auditeur expérimenté approfondirait une évaluation de conformité réelle.

# Pour chaque question, fournissez immédiatement après une réponse plausible et pertinente que l’organisation auditée pourrait donner, en lien avec le contexte décrit. Les réponses doivent être réalistes, refléter les bonnes pratiques, et s’aligner avec les exigences de la clause.

# Contexte :
# L’organisation auditée est une entreprise spécialisée dans les logiciels et les technologies. Elle comprend des équipes de développement logiciel ainsi que des équipes de validation. Ces équipes interagissent fréquemment avec le client télécom (opérateur) dans le monde entier, et collaborent avec des partenaires tiers pour la production des produits résidentiels. Les activités de développement, la validation du code, l’intégration et le déploiement continus, ainsi que la collaboration externe font partie intégrante de leur environnement opérationnel.

# L’entreprise utilise des outils comme JIRA, jenkins, Confluence, GitLab, SharePoint et Office 365 pour faciliter la gestion des tâches, la collaboration, et la gestion documentaire.
# les OS des postes de travail utilisés sont soit du Linux soit du Windows, les produits developpés sont avec des OS Android ou Linux selon le client en face.
# la structure de la société est faite de sorte à avoir des equipes de developpement , des equipes de validation , des equipes de certification fonctionnels, des equipes transverses  et un equipe infra reseau qui permet de fournir des environnements de test et de simulation des environnements client, ou des accées VPN aux infrastructures client( operateur)

# Consignes :
# - Commencez par une question générale pour évaluer la conformité de base.
# - Ensuite, posez des questions de plus en plus détaillées ou approfondies, en fonction des réponses probables.
# - Les questions doivent être spécifiques, exploitables et clairement liées au contenu de la clause et au contexte de l'entreprise.
# - Fournissez une réponse réaliste juste après chaque question qui se base sur les clauses de ISO comme référence, justifiant la réponse par rapport au document de Clause ISO fourni.
# - Adoptez un ton naturel et humain, tel qu’un auditeur experimentés dans le domaine de l'entreprise le ferait lors d’un entretien.
# - Incluez entre 3 et 7 paires questions/réponses, selon la complexité de la clause.
# - Assurez-vous que chaque question s’appuie sur la précédente — comme dans une conversation réfléchie ou une visite d’audit.
# - Formulez vos questions et réponses en tenant compte de l’environnement de l’entreprise : développement logiciel, validation, interactions clients et collaborations avec des tiers, ainsi que l’utilisation des outils mentionnés.
# - N’INCLUEZ AUCUNE mise en forme, puce ou explication — produisez uniquement les questions suivies de leur réponse sur deux lignes consécutives.

# Générez maintenant l’ensemble des paires questions/réponses en chaîne pour cette clause, en français. 

# """

#     response = gemma_model.invoke(prompt).content

#     # Split into lines, and group them as Q&A pairs
#     lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
#     qna_pairs = []

#     for i in range(0, len(lines) - 1, 2):
#         question = lines[i]
#         answer = lines[i + 1]
#         company_answer = ask_company_question(question)
#         qna_pairs.append({
#             "question": question,
#             "answer": answer,
#             "company_answer": company_answer
#         })

#     return {
#         "section": section,
#         "subsection": subsection,
#         "ref_id": ref,
#         "qna": qna_pairs,
#         "text": text  # Include the clause content
#     }

# def save_quiz_to_csv(quiz_results):
#     """
#     Save quiz results directly (without flattening) to a CSV-compatible JSON format.
#     """
#     import os
#     import json

#     filename = "iso27002_quiz_output_raw.json"  # Better suited for structured data
#     file_path = os.path.abspath(filename)

#     # Save directly as JSON for structured access later
#     with open(file_path, 'w', encoding='utf-8') as f:
#         json.dump(quiz_results, f, ensure_ascii=False, indent=2)

#     print(f"Raw quiz results saved to: {file_path}")
#     return file_path

    
    

# def display_quiz_in_streamlit(quiz_results):
#     """
#     Display quiz questions and answers in a Streamlit table with download buttons
#     """
#     import streamlit as st
#     import pandas as pd

#     st.markdown("## ISO 27002:2022 Audit Questions")

#     for i, item in enumerate(quiz_results):
#         section = item["section"]
#         subsection = item["subsection"]
#         ref_id = item["ref_id"]
#         qna = item["qna"]

#         with st.expander(f"{subsection}", expanded=(i == 0)):
#             st.markdown(f"**Section:** {section}")
#             st.markdown(f"**Reference:** {ref_id}")

#             questions_df = pd.DataFrame({
#                 "Question #": range(1, len(qna) + 1),
#                 "Audit Question": [q["question"] for q in qna],
#                 "Suggested Answer": [q["answer"] for q in qna],
#                 "Company Answer": [q["company_answer"] for q in qna]
#             })

#             st.table(questions_df)

#             # Download for individual clause
#             section_csv = questions_df.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label=f"📥 Download '{ref_id}' Questions",
#                 data=section_csv,
#                 file_name=f"iso27002_{ref_id}_questions.csv",
#                 mime="text/csv",
#                 key=f"download_section_{i}"
#             )

#     # Global CSV download
#     all_questions = []
#     for item in quiz_results:
#         for i, qna in enumerate(item["qna"]):
#             all_questions.append({
#                 "Section": item["section"],
#                 "Subsection": item["subsection"],
#                 "Reference ID": item["ref_id"],
#                 "Question #": i + 1,
#                 "Question": qna["question"],
#                 "Answer": qna["answer"],
#                 "Company Answer": qna["company_answer"]
#             })

#     all_questions_df = pd.DataFrame(all_questions)

#     if all_questions:
#         st.markdown("### Download All Questions")
#         full_csv = all_questions_df.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="📥 Download Complete Question Set",
#             data=full_csv,
#             file_name="iso27002_all_questions.csv",
#             mime="text/csv",
#             key="download_all"
#         )

# def generate_iso_quiz():
#     print("🔍 Generating ISO 27002 quiz questions...")

#     with open("Q27002_sheet.json", 'r', encoding='utf-8') as f:
#         structured_sections = json.load(f)

#     quiz_results = []

#     for item in structured_sections:
#         section = item.get("section")
#         subsections = item.get("subsections", [])

#         for subsection in subsections:
#             query = f"{subsection} {section}"
#             print(f"\n🔹 Searching for: {query}")

#             results = vector_store_iso_title.similarity_search(query=query, k=1)

#             if results:
#                 combined_context = ""
#                 for doc in results:
#                     title = doc.metadata.get("title", "")
#                     content = doc.metadata.get("content", "")
#                     source = doc.metadata.get("source", "")
#                     print(f"✅ Found title: {title}")
#                     print(f"📄 Content snippet: {content[:200]}...\n")

#                     combined_context += title + "\n" + content + "\n" + source + "\n"

#                 if combined_context.strip():
#                     quiz_item = generate_quiz_question(section, subsection, combined_context.strip())
#                     quiz_results.append(quiz_item)
#                 else:
#                     print(f"❌ No valid content for subsection: {subsection}")
#             else:
#                 print(f"❌ No matches found for: {subsection}")

#     return quiz_results


       
# # -----------------------------
# # LangGraph Router
# # -----------------------------

# print("Building main workflow graph...")
# class GraphState(TypedDict):
#     question: str
#     generation: str
#     documents: List
#     results: List

# workflow = StateGraph(GraphState)

# def gap_analysis_node(state: GraphState) -> GraphState:
#     result = run_gap_analysis()
#     result_text = f"Gap analysis completed."
    
#     # Get the actual DataFrame and CSV path
#     csv_path = result['csv']
#     results_df = result['df']
    
#     return {
#         "question": state["question"],
#         "generation": result_text,
#         "documents": [],
#         "results": results_df  # Store the actual results data
#     }
# def generate_quiz_node(state: GraphState) -> GraphState:
#     """
#     Generate quiz questions based on ISO 27002 sections.
#     Returns results and saves them as raw structured JSON.
#     """
#     # Generate the quiz
#     quiz_results = generate_iso_quiz()

#     # Count questions and sections for summary
#     sections_count = len(set([q['section'] for q in quiz_results]))
#     total_questions = sum(len(q['qna']) for q in quiz_results)

#     # Automatically save results in raw JSON format
#     quiz_file_path = save_quiz_to_csv(quiz_results)

#     # Create a detailed summary message
#     result_text = (
#         f"Quiz generation completed. Created {total_questions} questions "
#         f"across {sections_count} sections of ISO 27002:2022.\n\n"
#         f"Questions per clause vary based on complexity and content depth.\n"
#         f"Raw quiz results saved to: {quiz_file_path}"
#     )

#     return {
#         "question": state["question"],
#         "generation": result_text,
#         "documents": [],
#         "results": quiz_results
#     }

# def iso_qna_node(state: GraphState) -> GraphState:
#     # Get the question from state
#     question = state["question"]
#     # Run ISO Q&A
#     answer = ask_iso_question(question)
#     # Update state with answer
#     return {
#         "question": question,
#         "generation": answer,
#         "documents": [],  # Empty list as placeholder
#         "results": []
#     }
# def COMP_qna_node(state: GraphState) -> GraphState:
#     # Get the question from state
#     question = state["question"]
#     # Run COMP Q&A
#     answer = ask_company_question(question)
#     # Update state with answer
#     return {
#         "question": question,
#         "generation": answer,
#         "documents": [],  # Empty list as placeholder
#         "results": []
#     }

# def route_question(state: GraphState) -> str:
#     """
#     Route question to gap analysis, ISO Q&A, company docs, or quiz generation based on question and history.
#     """
#     question = state["question"]
    
#     # Get conversation history
#     history = ""
#     if "messages" in st.session_state:
#         # Format the last 5 messages (or fewer if not available)
#         history_msgs = st.session_state.messages[-5:] if len(st.session_state.messages) > 5 else st.session_state.messages
#         history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_msgs])
    
#     # Use the routing model to determine the destination
#     destination = route_chain.invoke({"query": question, "history": history})
#     print(f"📍 LLM Routing Destination: {destination}")
    
#     if destination == "gap_analysis":
#         return "run_gap_analysis"
#     elif destination == "company_docs":
#         return "ask_company_question"
#     elif destination == "quiz_generation":
#         return "generate_quiz"
#     else:
#         return "ask_iso_question"

# workflow.add_node("run_gap_analysis", gap_analysis_node)
# workflow.add_node("ask_iso_question", iso_qna_node)
# workflow.add_node("ask_company_question", COMP_qna_node)
# workflow.add_node("generate_quiz", generate_quiz_node)

# workflow.add_conditional_edges(START, route_question, {
#     "run_gap_analysis": "run_gap_analysis",
#     "ask_iso_question": "ask_iso_question",
#     "ask_company_question": "ask_company_question",
#     "generate_quiz": "generate_quiz"
# })
# workflow.add_edge("run_gap_analysis", END)
# workflow.add_edge("ask_iso_question", END)
# workflow.add_edge("ask_company_question", END)
# workflow.add_edge("generate_quiz", END)

# chat_graph = workflow.compile()
# def run_query_pipeline(user_input: str):
#     state = {"question": user_input, "generation": "", "documents": [], "results": []}
#     result = chat_graph.invoke(state)
#     return result["generation"], result["documents"], result["results"]

# # Ensure DB file exists
# def ensure_db_file_exists():
#     if not os.path.exists(DB_FILE):
#         with open(DB_FILE, 'w') as file:
#             json.dump({"chat_history": []}, file)
# def get_binary_file_downloader_html(bin_file, file_label='File'):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return data
# def main():
    
    
#     st.set_page_config(page_title="ISO 27002 Assistant", layout="centered")

#     st.title("🔍 ISO 27002 Assistant")
#     st.write("Ask about ISO 27002:2022, request a gap analysis, or just chat generally.")
    
#     # File uploader for CSV files
#     uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
#     for uploaded_file in uploaded_files:
#         bytes_data = uploaded_file.read()
#         st.write("filename:", uploaded_file.name)
#         st.write(bytes_data)
    
#     # Sidebar for model selection
#     st.sidebar.title("Model Settings")
    
#     # List of available models
#     models = ["gemini-1.5-pro", "llama-3.1-8b-instant"]
#     selected_model = st.sidebar.selectbox("Select Model", models, index=0)
    
#     # Set up the model based on selection
#     if selected_model == "gemini-1.5-pro":
#         st.session_state.current_model = "gemini"
#     else:
#         st.session_state.current_model = "llama"
    
#     # Ensure DB file exists
#     ensure_db_file_exists()
    
#     # Load chat history from db.json
#     with open(DB_FILE, 'r') as file:
#         db = json.load(file)
    
#     # Initialize messages in session state if not already present
#     if "messages" not in st.session_state:
#         st.session_state.messages = db.get('chat_history', [])

#     # Display chat messages from history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat UI setup
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     # Input box
#     user_input = st.chat_input("Enter your question or request:")

#     if user_input:
#         # Add user message to both chat formats
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         st.session_state.history.append({"role": "user", "text": user_input})
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         # Process the query using our pipeline
#         with st.spinner("Processing your request..."):
#             result, docs, results_data = run_query_pipeline(user_input)
        
#         # Display assistant response
#         with st.chat_message("assistant"):
#             st.markdown(result)
#             # Handle gap analysis results (if any)
#             if results_data and isinstance(results_data, list) and results_data and "subsection" in results_data[0]:
#                 # This is quiz data, display it using the specialized quiz display function
#                 display_quiz_in_streamlit(results_data)
#             elif results_data:
#                 st.markdown("### Gap Analysis Results")
                
#                 # Convert to DataFrame for display
#                 results_df = pd.DataFrame(results_data)
                
#                 # Display the results as a table
#                 st.dataframe(results_df[["theme", "clause", "alignment_status","gap_analysis","recommendations","iso_source"]])
                
#                 # Create a CSV download button using the DataFrame
#                 csv = results_df.to_csv(index=False)
#                 st.download_button(
#                     label="📥 Download Gap Analysis CSV",
#                     data=csv,
#                     file_name="gap_analysis.csv",
#                     mime="text/csv",
#                 )
#                 docx_path = save_to_docx(results_data)
    
#                 if os.path.exists(docx_path):
#                     with open(docx_path, "rb") as docx_file:
#                         docx_data = docx_file.read()
#                         st.download_button(
#                             label="📄 Download Gap Analysis DOCX",
#                             data=docx_data,
#                             file_name="gap_analysis.docx",
#                             mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
#                         )
#                 else:
#                     st.warning(f"DOCX file could not be generated")
            
#             # Handle document display for regular Q&A
#             if docs:
#                 with st.expander("📄 Retrieved Documents"):
#                     for doc in docs:
#                         if isinstance(doc, str):
#                             st.markdown(f"**Source:** {doc}")
#                             st.text("Document content not available")
#                         elif hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
#                             st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
#                             st.text(doc.page_content)
#                         else:
#                             st.markdown(f"**Document:** {str(doc)}")
        
#         # Add assistant response to messages
#         st.session_state.messages.append({"role": "assistant", "content": result})
#         st.session_state.history.append({
#             "role": "ai", 
#             "text": result, 
#             "docs": docs if isinstance(docs, list) else [],
#             "results": results_data if isinstance(results_data, list) else []
#         })
        
#         # Store chat history to db.json
#         db['chat_history'] = st.session_state.messages
#         with open(DB_FILE, 'w') as file:
#             json.dump(db, file)

#     # Add a "Clear Chat" button to the sidebar
#     if st.sidebar.button('Clear Chat'):
#         # Clear chat history in db.json
#         db['chat_history'] = []
#         with open(DB_FILE, 'w') as file:
#             json.dump(db, file)
        
#         # Clear chat messages in session state
#         st.session_state.messages = []
#         st.session_state.history = []
#         st.rerun()

# if __name__ == "__main__":
#     main()