import streamlit as st
import os
import json
from typing import List, TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
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
from typing import List, TypedDict, Literal
from langchain_core.messages import HumanMessage
from node.company_docs import ask_company_question
from node.iso_qna import ask_iso_question
from node.quiz_iso27002_company import generate_iso_quiz, save_quiz_to_csv, display_quiz_in_streamlit
from node.generate_document import load_json_template,create_document_from_template
from node.gap_analysis import run_gap_analysis,display_gap_analysis_in_streamlit
from node.quiz_iso42001 import generate_iso_quiz_2, save_quiz_to_csv_2,display_quiz_in_streamlit_2
from node.quiz_iso27002_human_eval import generate_iso_quiz_1, display_quiz_in_streamlit_1, evaluate_quiz_responses_1, display_evaluation_results_1
print("Setting up graphs and language models...")

# Define the database file path for chat history
DB_FILE = "db.json"



# -----------------------------
# Initialize Language Models
# -----------------------------
print("Initializing language models...")



llm_ollama =ChatOllama(model="qwen3:0.6b",temperature=0.7)

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
#Graph

print("Setting up routing logic...")
class RouteQuery(TypedDict):
    destination: Literal[
        "iso_qna",
        "gap_analysis",
        "company_docs",
        "quiz_iso27002_company",
        "quiz_iso27002_human_eval",
        "quiz_iso42001",
        "generate_document"
    ]

route_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant.
Determine if the user is asking about:
1. ISO 27001 or ISO 27002 or ISO 42001 standard information (iso_qna)
2. A gap analysis request (gap_analysis)
3. Company-specific document information (company_docs)
4. Generate an ISO 27002 quiz with company answers (quiz_iso27002_company)
5. Generate an ISO 27002 quiz with human answers and evaluation (quiz_iso27002_human_eval)
6. Generate an ISO 42001 quiz (quiz_iso42001)
7. Document generation from template request (generate_document)

When a user asks about:
- ISO 27001, ISO 27002, or ISO 42001 standards, route to iso_qna .
- Creating a document, generating a report based on a template, or mentions document templates or JSON templates, route to generate_document.
- Generating a quiz for ISO 27002 with company answers, route to quiz_iso27002_company.
- Generating a quiz for ISO 27002 with human answers and evaluation, route to quiz_iso27002_human_eval.
- Generating a quiz for ISO 42001, route to quiz_iso42001.

Consider the entire conversation history when making your decision.

Respond strictly in JSON format as follows:
{{"destination": "iso_qna"}}  or 
{{"destination": "gap_analysis"}} or {{"destination": "company_docs"}} or 
{{"destination": "quiz_iso27002_company"}} or {{"destination": "quiz_iso27002_human_eval"}} or 
{{"destination": "quiz_iso42001"}} or {{"destination": "generate_document"}}"""),
    ("human", "{query}"),
    ("human", "Previous conversation: {history}")
])

route_chain = (
    route_prompt
    | gemma_model.with_structured_output(RouteQuery, include_raw=False)
    | itemgetter("destination")
)
# -----------------------------
# LangGraph Router
# -----------------------------

print("Building main workflow graph...")
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List
    results: List

workflow = StateGraph(GraphState)
# Update the document generation node to work with the default AIMS.json file
def generate_document_node(state: GraphState) -> GraphState:
    """Generate a document based on the provided template file."""
    try:
        # Default to the new template file
        file_path = r"C:\Users\G800613RTS\Desktop\Chatbot\Templates\AI Management System Policy.json"
        
        print(f"Loading template from: {file_path}")
        
        # Load the JSON template
        template_json = load_json_template(file_path)
        
        # Run the document generation process
        result = create_document_from_template(template_json)
        
        # Get PDF data
        pdf_data = result["pdf_data"]
        json_filename = template_json["document_details"]["document_name"].replace(" ", "_")
        pdf_filename = f"{json_filename}.pdf"
        
        # Save PDF to file
        with open(pdf_filename, "wb") as f:
            f.write(pdf_data)
        
        # Prepare detailed generation report
        section_info = []
        for section in result["generated_sections"]:
            section_title = section["title"]
            content_preview = section["content"][:50] + "..." if len(section["content"]) > 50 else section["content"]
            section_info.append(f"- {section_title}: {content_preview}")
        
        section_report = "\n".join(section_info)
        
        # Prepare response
        result_text = f"""# Document Generation Complete

I've created the **{template_json['document_details']['document_name']}** document with {len(result['generated_sections'])} sections based on the AI Policy template.

## Document Information
- **Document ID:** {template_json['document_details']['document_id']}
- **Version:** {template_json['document_details']['version']}
- **Date:** {template_json['document_details']['date']}
- **Author:** {template_json['document_details']['author']}
- **Approved by:** {template_json['document_details']['approver']}

## Generated Sections Overview
{len(template_json['policy_sections'])} policy sections were processed and reviewed for ISO 42001 compliance.

The PDF has been saved as: {pdf_filename}

You can download the document using the button below.
"""
        
        return {
            "question": state["question"],
            "generation": result_text,
            "documents": [],
            "results": {
                "pdf_path": pdf_filename,
                "pdf_data": pdf_data,
                "document_title": template_json['document_details']['document_name']
            }
        }
    except Exception as e:
        error_message = f"Error generating document: {str(e)}"
        print(error_message)
        return {
            "question": state["question"],
            "generation": error_message,
            "documents": [],
            "results": []
        }

def gap_analysis_node(state: GraphState) -> GraphState:
    result = run_gap_analysis()
    result_text = f"Gap analysis completed."
    
    # Get the actual DataFrame and CSV path
    csv_path = result['csv']
    results_df = result['df']
    
    return {
        "question": state["question"],
        "generation": result_text,
        "documents": [],
        "results": results_df  # Store the actual results data
    }
def generate_quiz_node_2(state: GraphState) -> GraphState:
    """
    Generate quiz questions based on ISO 27002 sections.
    Returns results and saves them as raw structured JSON.
    """
    # Generate the quiz
    quiz_results = generate_iso_quiz_2()

    # Count questions and sections for summary
    sections_count = len(set([q['section'] for q in quiz_results]))
    total_questions = sum(len(q['qna']) for q in quiz_results)

    # Automatically save results in raw JSON format
    quiz_file_path = save_quiz_to_csv_2(quiz_results)

    # Create a detailed summary message
    result_text = (
        f"Quiz generation completed. Created {total_questions} questions "
        f"across {sections_count} sections of ISO 42001:2023.\n\n"
        f"Questions per clause vary based on complexity and content depth.\n"
        f"Raw quiz results saved to: {quiz_file_path}"
    )

    return {
        "question": state["question"],
        "generation": result_text,
        "documents": [],
        "results": quiz_results
    }
def generate_quiz_node_1(state: GraphState) -> GraphState:
    """
    Generate quiz questions based on ISO 27002 sections.
    Returns results without saving them to a file.
    """
    # Generate the quiz
    quiz_results = generate_iso_quiz_1()


    # Count questions and sections for summary
    sections_count = len(set([q['section'] for q in quiz_results]))
    total_questions = sum(len(q['qna']) for q in quiz_results)

    # Create a detailed summary message
    result_text = (
        f"Quiz generation completed. Created {total_questions} questions "
        f"across {sections_count} sections of ISO 27002:2022.\n\n"
        f"Questions per clause vary based on complexity and content depth."
    )

    return {
        "question": state["question"],
        "generation": result_text,
        "documents": [],
        "results": quiz_results
    }
def generate_quiz_node(state: GraphState) -> GraphState:
    """
    Generate quiz questions based on ISO 27002 sections.
    Returns results and saves them as raw structured JSON.
    """
    # Generate the quiz
    quiz_results = generate_iso_quiz()

    # Count questions and sections for summary
    sections_count = len(set([q['section'] for q in quiz_results]))
    total_questions = sum(len(q['qna']) for q in quiz_results)

    # Automatically save results in raw JSON format
    quiz_file_path = save_quiz_to_csv(quiz_results)

    # Create a detailed summary message
    result_text = (
        f"Quiz generation completed. Created {total_questions} questions "
        f"across {sections_count} sections of ISO 27002:2022.\n\n"
        f"Questions per clause vary based on complexity and content depth.\n"
        f"Raw quiz results saved to: {quiz_file_path}"
    )

    return {
        "question": state["question"],
        "generation": result_text,
        "documents": [],
        "results": quiz_results
    }

def iso_qna_node(state: GraphState) -> GraphState:
    """
    Process ISO standard questions using the LangGraph implementation.
    Uses the graph from iso_qna.py which provides memory persistence and tool selection.
    """
    # Get the question from state
    question = state["question"]
    
    # Generate a thread_id for this conversation if not available
    # For simplicity, we'll use a hash of the question content
    thread_id = f"iso-thread-{hash(question) % 10000:04d}"
    
    # Initialize session state for ISO conversation context if not present
    if "iso_conversation_context" not in st.session_state:
        st.session_state.iso_conversation_context = {}
    
    # Create a key for this thread in the conversation context
    thread_key = thread_id
    if thread_key not in st.session_state.iso_conversation_context:
        st.session_state.iso_conversation_context[thread_key] = {
            "sources": [],
            "last_question": "",
            "last_answer": ""
        }
    
    # Check if this is likely a follow-up question (short query)
    is_followup = len(question.split()) <= 5
    
    # Use the graph from iso_qna.py with the appropriate thread_id for memory persistence
    from node.iso_qna import graph
    
    # Configure the graph with the thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    # Invoke the graph with the user's question
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    
    # Extract the answer from the graph's response
    if result and "messages" in result and result["messages"]:
        # Get the last message which should be the AI's response
        answer = result["messages"][-1].content
        
        # Extract sources from the tools used in the graph (if available)
        new_sources = []
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        
        # If we have tool messages, extract source information
        if tool_messages:
            for tool_msg in tool_messages:
                if hasattr(tool_msg, "artifact") and tool_msg.artifact:
                    # Extract documents from the tool's artifact
                    docs = tool_msg.artifact
                    for doc in docs:
                        if hasattr(doc, "metadata") and doc.metadata:
                            # Add source information to our sources list
                            source = doc.metadata.get('source', 'Unknown')
                            if source not in new_sources:
                                new_sources.append(source)
            
            # Store new sources for this conversation if we found any
            if new_sources:
                st.session_state.iso_conversation_context[thread_key]["sources"] = new_sources
                st.session_state.iso_conversation_context[thread_key]["last_question"] = question
                st.session_state.iso_conversation_context[thread_key]["last_answer"] = answer
                
                # Use the new sources for this response
                sources_to_return = new_sources
            else:
                # No new sources, use previous sources if this is a follow-up
                sources_to_return = st.session_state.iso_conversation_context[thread_key]["sources"]
        else:
            # No tool messages, use previous sources if available
            sources_to_return = st.session_state.iso_conversation_context[thread_key]["sources"]
        
        # Save the current Q&A pair
        st.session_state.iso_conversation_context[thread_key]["last_question"] = question
        st.session_state.iso_conversation_context[thread_key]["last_answer"] = answer
        
        # Return the response with appropriate sources
        return {
            "question": question,
            "generation": answer,
            "documents": sources_to_return,
            "results": [],
        }
    else:
        # Fallback in case the graph doesn't return expected results
        # Still try to use previous sources if available
        previous_sources = st.session_state.iso_conversation_context[thread_key]["sources"] if thread_key in st.session_state.iso_conversation_context else []
        
        return {
            "question": question,
            "generation": "I'm sorry, I couldn't process your ISO standard question at this time.",
            "documents": previous_sources,
            "results": [],
        }
    
def COMP_qna_node(state: GraphState) -> GraphState:
    # Get the question from state
    question = state["question"]
    
    # Run COMP Q&A which returns (answer, sources)
    answer, sources = ask_company_question(question)
    
    # Update state with answer and sources
    return {
        "question": question,
        "generation": answer,  # Store just the text response in generation
        "documents": sources,  # Store the sources in documents
        "results": []
    }

def route_question(state: GraphState) -> str:
    """
    Route question to appropriate node based on question and history.
    """
    question = state["question"]
    
    # Get conversation history
    history = ""
    if "messages" in st.session_state:
        # Format the last 5 messages (or fewer if not available)
        history_msgs = st.session_state.messages[-5:] if len(st.session_state.messages) > 5 else st.session_state.messages
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_msgs])
    
    # Use the routing model to determine the destination
    destination = route_chain.invoke({"query": question, "history": history})
    print(f"📍 LLM Routing Destination: {destination}")
    
    # Map the destination to the appropriate node
    if destination == "iso_qna":
        
        return "ask_iso_question"
    elif destination == "gap_analysis":
        return "run_gap_analysis"
    elif destination == "company_docs":
        return "ask_company_question"
    elif destination == "quiz_iso27002_company":
        return "generate_quiz_company"
    elif destination == "quiz_iso27002_human_eval":
        return "generate_quiz_human_eval"
    elif destination == "quiz_iso42001":
        return "generate_quiz_iso42001"
    elif destination == "generate_document":
        return "generate_document"
    else:
        # Default to ISO 27002 Q&A for unrecognized destinations
        state["iso_standard"] = "27002"
        return "ask_iso_question"

workflow.add_node("run_gap_analysis", gap_analysis_node)
workflow.add_node("ask_iso_question", iso_qna_node)
workflow.add_node("ask_company_question", COMP_qna_node)
workflow.add_node("generate_quiz_company", generate_quiz_node)
workflow.add_node("generate_quiz_human_eval", generate_quiz_node_1)
workflow.add_node("generate_quiz_iso42001", generate_quiz_node_2)
workflow.add_node("generate_document", generate_document_node)
workflow.add_conditional_edges(START, route_question, {
    "run_gap_analysis": "run_gap_analysis",
    "ask_iso_question": "ask_iso_question",
    "ask_company_question": "ask_company_question",
    "generate_quiz_company": "generate_quiz_company",
    "generate_quiz_human_eval": "generate_quiz_human_eval",
    "generate_quiz_iso42001": "generate_quiz_iso42001",
    "generate_document": "generate_document"
})
workflow.add_edge("run_gap_analysis", END)
workflow.add_edge("ask_iso_question", END)
workflow.add_edge("ask_company_question", END)
workflow.add_edge("generate_quiz_company", END)
workflow.add_edge("generate_quiz_human_eval", END)
workflow.add_edge("generate_quiz_iso42001", END)
workflow.add_edge("generate_document", END)
chat_graph = workflow.compile()
def run_query_pipeline(user_input: str):
    state = {"question": user_input, "generation": "", "documents": [], "results": []}
    result = chat_graph.invoke(state)
    return result["generation"], result["documents"], result["results"]

# Ensure DB file exists
def ensure_db_file_exists():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as file:
            json.dump({"chat_history": []}, file)
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return data
def main():
    
    
    st.set_page_config(page_title="ISO Assistant", layout="centered")

    st.title("🔍 ISO Assistant")
    st.write("Ask about ISO Standard, request a gap analysis, or just chat generally.")
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    
    # List of available models
    models = ["gemma3:12b-it-qat", "deepseek-r1:7b"]
    selected_model = st.sidebar.selectbox("Select Model", models, index=0)
    
    # Set up the model based on selection
    if selected_model == "deepseek-r1:7b":
        st.session_state.current_model = "gemma3:12b-it-qat"
    else:
        st.session_state.current_model = "deepseek"
    
    # st.sidebar.markdown("#### PDF Files")
    # pdf_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])
    
    # if pdf_file:
    #     # Create directory if it doesn't exist
    #     os.makedirs(UPLOADS_DIR, exist_ok=True)
        
    #     # Save the uploaded file to the working directory
    #     file_path = f"{UPLOADS_DIR}/{pdf_file.name}"
    #     with open(file_path, "wb") as f:
    #         f.write(pdf_file.getbuffer())
        
    #     st.sidebar.success(f"PDF saved successfully: {pdf_file.name}")
        
    #     # Process button
    #     if st.sidebar.button("Process PDF & Update Vector Store"):
    #         with st.spinner("Processing PDF..."):
    #             try:
    #                 # Define the Qwen model path
    #                 qwen_model_path = r"C:\Models\Qwen2.5-VL-7B-Instruct"
                    
    #                 # Process the PDF
    #                 st.info("Step 1/3: Extracting text from PDF using Qwen...")
    #                 chunks, output_json = process_pdf(file_path, qwen_model_path)
                    
    #                 # Summarize chunks
    #                 st.info("Step 2/3: Summarizing chunks...")
    #                 summarized_chunks = summarize_chunks(chunks, pdf_file.name)
                    
    #                 # Update vector stores
    #                 st.info("Step 3/3: Updating vector stores...")
    #                 summary_count, original_count = update_vector_stores(summarized_chunks)
                    
    #                 st.success(f"✅ Successfully processed PDF and added {summary_count} documents to vector stores!")
                    
    #             except Exception as e:
    #                 st.error(f"Error processing PDF: {str(e)}")
    
    ensure_db_file_exists()
    
    # Load chat history from db.json
    with open(DB_FILE, 'r') as file:
        db = json.load(file)
    
    # Initialize messages in session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = db.get('chat_history', [])
    
    # Initialize document generation results if not present
    if "document_results" not in st.session_state:
        st.session_state.document_results = None
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Check if a quiz is in session and display it based on quiz type
    if "current_quiz" in st.session_state and "quiz_type" in st.session_state:
        with st.chat_message("assistant"):
            # Display the quiz based on its type
            if st.session_state.quiz_type == "quiz_iso27002_company":
                display_quiz_in_streamlit(st.session_state.current_quiz)
            elif st.session_state.quiz_type == "quiz_iso27002_human_eval":
                display_quiz_in_streamlit_1(st.session_state.current_quiz)
                
                # Add separate evaluate button outside the quiz display
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🚀 Evaluate All Responses", key="evaluate_btn_main"):
                        # Evaluate responses and get results
                        evaluation_results = evaluate_quiz_responses_1()
                        if evaluation_results:
                            # Store results in session state
                            st.session_state.evaluation_results = evaluation_results
                            st.rerun()  # Force refresh to display results
                
                # Display evaluation results if they exist
                if "evaluation_results" in st.session_state and st.session_state.evaluation_results:
                    display_evaluation_results_1(st.session_state.evaluation_results)
            elif st.session_state.quiz_type == "quiz_iso42001":
                display_quiz_in_streamlit_2(st.session_state.current_quiz)
    
    # Display gap analysis if in session
    if "current_gap_analysis" in st.session_state:
        with st.chat_message("assistant"):
            display_gap_analysis_in_streamlit(st.session_state.current_gap_analysis)
    
    # Chat UI setup
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # Display document download button if a document was generated
    if st.session_state.document_results:
        st.markdown("### Generated Document")
        st.markdown(f"**Title:** {st.session_state.document_results['document_title']}")
        
        # Download button for PDF
        st.download_button(
            label="📄 Download Generated Document",
            data=st.session_state.document_results["pdf_data"],
            file_name=f"{st.session_state.document_results['document_title'].replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

    # Input box
    user_input = st.chat_input("Enter your question or request:")

    if user_input:
        # Add user message to both chat formats
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "user", "text": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process the query using our pipeline
        with st.spinner("Processing your request..."):
            result, docs, results_data = run_query_pipeline(user_input)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(result)
            
            # Handle different types of results based on content and route destination
            if isinstance(results_data, dict) and "pdf_data" in results_data:
                # Document generation results
                st.session_state.document_results = results_data
                
                # Add download button for immediate use
                st.download_button(
                    label="📄 Download Generated Document",
                    data=results_data["pdf_data"],
                    file_name=f"{results_data['document_title'].replace(' ', '_')}.pdf",
                    mime="application/pdf",
                )
            elif isinstance(results_data, list) and results_data:
                if all(isinstance(item, dict) for item in results_data):
                    # Determine the type of data based on its structure
                    if "subsection" in results_data[0]:
                        # Check for quiz data type markers to determine which quiz display to use
                        if "qna1" in results_data[0]:
                            # Human evaluation quiz (ISO 27002)
                            st.session_state.current_quiz = results_data
                            st.session_state.quiz_type = "quiz_iso27002_human_eval"
                            display_quiz_in_streamlit_1(results_data)
                            
                            # Add separate evaluate button outside the quiz display
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if st.button("🚀 Evaluate All Responses", key="evaluate_btn_inline"):
                                    # Evaluate responses and get results
                                    evaluation_results = evaluate_quiz_responses_1()
                                    if evaluation_results:
                                        # Store results in session state
                                        st.session_state.evaluation_results = evaluation_results
                                        st.rerun()  # Force refresh to display results
                            
                            # Display evaluation results if they exist
                            if "evaluation_results" in st.session_state and st.session_state.evaluation_results:
                                display_evaluation_results_1(st.session_state.evaluation_results)
                        elif "company_answer" in results_data[0]["qna"][0]:
                            # Company-focused quiz (ISO 27002)
                            st.session_state.current_quiz = results_data
                            st.session_state.quiz_type = "quiz_iso27002_company"
                            display_quiz_in_streamlit(results_data)
                        elif "qna" in results_data[0] and len(results_data[0]["qna"]) > 0:
                            # ISO 42001 quiz
                            st.session_state.current_quiz = results_data
                            st.session_state.quiz_type = "quiz_iso42001"
                            display_quiz_in_streamlit_2(results_data)
                    elif "theme" in results_data[0] and "clause" in results_data[0]:
                        # Gap analysis results
                        st.session_state.current_gap_analysis = results_data
                        display_gap_analysis_in_streamlit(results_data)
            
            # Handle document display for regular Q&A
            if docs:
                with st.expander("📄 Retrieved Documents"):
                    for doc in docs:
                        if isinstance(doc, str):
                            st.markdown(f"**Source:** {doc}")
                        elif hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                            st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.text(doc.page_content)
                        else:
                            st.markdown(f"**Document:** {str(doc)}")
        
        # Add assistant response to messages
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.session_state.history.append({
            "role": "ai", 
            "text": result, 
            "docs": docs if isinstance(docs, list) else [],
            "results": results_data if isinstance(results_data, list) else []
        })
        
        # Store chat history to db.json
        db['chat_history'] = st.session_state.messages
        with open(DB_FILE, 'w') as file:
            json.dump(db, file)

    # Add a "Clear Chat" button to the sidebar
    if st.sidebar.button('Clear Chat'):
        # Clear chat history in db.json
        db['chat_history'] = []
        with open(DB_FILE, 'w') as file:
            json.dump(db, file)
        
        # Clear chat messages in session state
        st.session_state.messages = []
        st.session_state.history = []
        # Clear quiz data
        if "current_quiz" in st.session_state:
            del st.session_state.current_quiz
        if "quiz_responses" in st.session_state:
            del st.session_state.quiz_responses
        if "evaluation_results" in st.session_state:
            del st.session_state.evaluation_results
            
        # Clear gap analysis data
        if "current_gap_analysis" in st.session_state:
            del st.session_state.current_gap_analysis
        if "gap_analysis_feedback" in st.session_state:
            del st.session_state.gap_analysis_feedback
        if "updated_gap_analysis" in st.session_state:
            del st.session_state.updated_gap_analysis
        st.rerun()

if __name__ == "__main__":
    main()