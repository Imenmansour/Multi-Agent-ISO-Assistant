import os
import json
from typing import List, Dict, TypedDict, Optional
from langgraph.graph import END, START, StateGraph
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
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
from typing import List, Dict, TypedDict, Optional
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

# -----------------------------
# Document Generation System
# -----------------------------
class DocumentState(TypedDict):
    template: Dict
    current_section_index: int
    generated_sections: List[Dict]
    section_content: str
    section_feedback: str
    is_approved: bool
    complete: bool
    pdf_data: Optional[bytes]
    current_documents: List
    regeneration_attempts: int

# Document generation graph
document_graph = StateGraph(DocumentState)

def document_generator_node(state: DocumentState) -> DocumentState:
    """Generate content for the current section based on ISO documentation."""
    print("document generator : ", state.get("regeneration_attempt",None))
    template = state["template"]
    current_index = state["current_section_index"]
    regeneration_attempts = state.get("regeneration_attempts", None)
    section_feedback = state.get("section_feedback", "")
    
    # If processing cover page
    if current_index == -1:
        doc_details = template["document_details"]
        content = f"""# {doc_details['document_name']}
### {doc_details['english_name']}

**Document ID:** {doc_details['document_id']}
**Version:** {doc_details['version']}
**Date:** {doc_details['date']}
**Author:** {doc_details['author']}
**Approved by:** {doc_details['approver']}

## Purpose
{doc_details['purpose']['description']}
"""
        return {
            **state,
            "section_content": content,
            "current_documents": []
        }
    
    # If processing table of contents
    if current_index == 0:
        policy_sections = template["policy_sections"]
        toc = "# Table of Contents\n\n"
        
        for idx, section in enumerate(policy_sections, 1):
            section_num = section.get("section_number", str(idx))
            french_title = section.get("french_title", f"Section {section_num}")
            toc += f"{section_num}. {french_title}\n"
        
        return {
            **state,
            "section_content": toc,
            "current_documents": []
        }
    
    # If processing a regular section (adjust index to account for TOC)
    adjusted_idx = current_index - 1
    if adjusted_idx < len(template["policy_sections"]):
        section = template["policy_sections"][adjusted_idx]
        section_num = section.get("section_number", str(adjusted_idx + 1))
        french_title = section.get("french_title", f"Section {section_num}")
        english_description = section.get("english_description", "")
        relevant_clauses = section.get("relevant_iso_clauses", [])
        
        # Log appropriate message based on whether this is initial generation or regeneration
        if regeneration_attempts!=None and regeneration_attempts > 0:
            print(f"Regenerating content for '{section_num}: {french_title}' (attempt {regeneration_attempts})")
        else:
            print(f"Generating initial content for '{section_num}: {french_title}'")
        
        # Create query combining section info and ISO clauses
        iso_clause_text = " ".join(relevant_clauses)
        query = f"{french_title} {english_description} ISO 42001 {iso_clause_text}"
        
        # Retrieve relevant ISO documents based on clauses
        documents = []
        for clause in relevant_clauses:
            clause_query = f"ISO 42001 clause {clause}"
            clause_docs = vector_store_42001_originals.similarity_search(clause_query, k=2)
            documents.extend(clause_docs)
        
        # Add general section documents
        section_docs = vector_store_42001_originals.similarity_search(query, k=3)
        documents.extend(section_docs)
        
        # Rerank all documents
        reranked_docs = rerank_documents(query, documents, top_k=3)
        
        # Format context for generation
        context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Title: {doc.metadata.get('title', 'No title')}\n"
            f"Content: {doc.page_content}\n"
            f"ISO Clause: {', '.join(relevant_clauses)}"
            for doc in reranked_docs
        )
        
        # Initialize feedback instructions (empty for first generation)
        feedback_instructions = ""
        
        # Only include feedback instructions if this is a regeneration attempt with feedback
        if regeneration_attempts!=None and regeneration_attempts > 0 and section_feedback:
            previous_content = state.get("section_content", "")
            feedback_instructions = f"""
PREVIOUS CONTENT THAT NEEDS IMPROVEMENT:
{previous_content}

REVIEWER FEEDBACK:
{section_feedback}

Please revise the content based on the above feedback. Focus specifically on addressing the issues 
identified by the reviewer while maintaining alignment with ISO 42001 requirements.
Make sure to fix any problems mentioned in the feedback while following the ISO clauses closely.
"""
        
        # Generate content with LLM - include feedback instructions only when applicable
        prompt = f"""
You are an ISO 42001 documentation specialist. Generate content for the following section of an AI Management System policy document:

Section Number: {section_num}
Section Title (French): {french_title}
Section Description (English): {english_description}
Relevant ISO 42001:2023 Clauses: {', '.join(relevant_clauses)}

{feedback_instructions}

Use the following ISO 42001 context to inform your writing:
{context}

The content should:
1. Be written in French to match the document's language
2. Be professionally written and ISO-compliant
3. Include appropriate subheadings and structure
4. Be comprehensive but concise (about 1-2 pages worth of content)
5. Include specific requirements relevant to this section
6. Reference the ISO clauses naturally in the text
7. Use proper terminology from ISO 42001:2023

Format your response as clean Markdown, starting with a level 2 heading for the section title. Don't use ** characters.
"""
        
        # Generate content using the LLM
        # content = llm_ollama.invoke(prompt).content
        content = "Generated content goes here."  # Placeholder for generated content
        
        # Add section number and title if not already included
        if not content.strip().startswith("#"):
            content = f"## {section_num}. {french_title}\n\n{content}"
        print({**state,
            "regeneration_attempts": regeneration_attempts
        }["regeneration_attempts"])
        return {
            **state,
            # "section_content": content,
            "section_content": "lol",
            "current_documents": reranked_docs
        }
    
    # If all sections are processed, mark as complete
    return {
        **state,
        "complete": True,
        "section_content": "Document generation complete."
    }
def document_reviewer_node(state: DocumentState) -> DocumentState:
    """Review the generated content for accuracy and completeness."""
    print("document reviewer : ", state.get("regeneration_attempt",None))
    template = state["template"]
    current_index = state["current_section_index"]
    content = state["section_content"]
    documents = state["current_documents"]
    
    # Initialize regeneration attempts if not present
    regeneration_attempts = state.get("regeneration_attempts",0)
    
    # If processing cover page (auto-approve)
    if current_index == -1:
        print("Cover page reviewed and approved")
        # Add cover page to generated sections
        generated_sections = state["generated_sections"].copy()
        generated_sections.append({
            "title": "Cover Page",
            "content": content
        })
        
        # Move to table of contents and reset regeneration counter
        
        return {
            **state,
            "generated_sections": generated_sections,
            "is_approved": True,
            "section_feedback": "Cover page approved.",
            "current_section_index": current_index + 1,
            "regeneration_attempts": 0
        }
    
    # If processing table of contents (auto-approve)
    if current_index == 0:
        print("Table of contents reviewed and approved")
        # Add TOC to generated sections
        generated_sections = state["generated_sections"].copy()
        generated_sections.append({
            "title": "Table of Contents",
            "content": content
        })
        
        # Move to first regular section and reset regeneration counter
        return {
            **state,
            "generated_sections": generated_sections,
            "is_approved": True,
            "section_feedback": "Table of contents approved.",
            "current_section_index": current_index + 1,
            "regeneration_attempts": 0
        }
    
    # If all sections are complete (account for TOC)
    adjusted_idx = current_index - 1
    if adjusted_idx >= len(template["policy_sections"]):
        print("All sections reviewed, proceeding to finalization")
        return {
            **state,
            "is_approved": True,
            "section_feedback": "All sections approved.",
            "complete": True,
            "regeneration_attempts": 0
        }
    
    # For regular sections, review against ISO documentation
    section = template["policy_sections"][adjusted_idx]
    section_num = section.get("section_number", str(adjusted_idx + 1))
    french_title = section.get("french_title", f"Section {section_num}")
    english_description = section.get("english_description", "")
    relevant_clauses = section.get("relevant_iso_clauses", [])
    
    print(f"Reviewing section {section_num}: {french_title}")
    
    # Format context for review
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Title: {doc.metadata.get('title', 'No title')}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )
    
    # Review content with LLM
    prompt = f"""
You are an ISO compliance reviewer checking a section of an AI Management System policy document.

Section Number: {section_num}
Section Title (French): {french_title}
Section Description (English): {english_description}
Relevant ISO 42001:2023 Clauses: {', '.join(relevant_clauses)}

Generated Content:
{content}

ISO 42001 Reference Material:
{context}

Please review the generated content and evaluate:
1. Accuracy - Does it comply with ISO 42001:2023 requirements, especially clauses {', '.join(relevant_clauses)}?
2. Completeness - Does it address all necessary aspects described in the section description?
3. Structure - Is it well-organized and professional?
4. Language - Is it written in clear, professional French?
5. Compliance - Does it meet ISO standards and use proper terminology?

Return your assessment in this format:
- APPROVED with a pourcentage of confidence (e.g., 95%) if it <50% will considarte as not approved
- Feedback: Your detailed feedback here
- Suggested Improvements: Specific suggestions for improvement if not approved
"""
    
    # Get review from LLM
    # review_result = llm_ollama.invoke(prompt).content
    # print(f"Review result: {review_result}")
    # Parse the result to determine if approved
    # Parse the result to extract the percentage and determine if approved
    # percentage_match = re.search(r'APPROVED.+?(\d+)%', review_result, re.IGNORECASE)
    
    # Default to not approved if no percentage found
    # confidence_percentage = 0
    # if percentage_match:
    #     confidence_percentage = int(percentage_match.group(1))
    confidence_percentage=0
    
    is_approved = confidence_percentage >= 70  # Approve only if 70% or higher
    
    if is_approved:
        print(f"Section '{section_num}: {french_title}' approved")
        # Add approved content to generated sections
        generated_sections = state["generated_sections"].copy()
        generated_sections.append({
            "title": f"{section_num}. {french_title}",
            "content": content                                                                                                                                                                                                                                                                              
        })
        
        # Move to next section and reset regeneration counter
        next_index = current_index + 1
        
        return {
            **state,
            "generated_sections": generated_sections,
            "is_approved": True,
            "section_feedback": f"Section approved: {section_num}. {french_title}. Moving to next section.",
            "current_section_index": next_index,
            "regeneration_attempts": 0
        }
    else:
        # Check if we've reached the maximum regeneration attempts
        regeneration_attempts += 1
        print(f"Section '{section_num}: {french_title}' not approved, will regenerate content (attempt {regeneration_attempts}/5)")
        
        if regeneration_attempts >= 5:
            print(f"⚠️ Maximum regeneration attempts reached for section '{french_title}'. Using current content and moving on.")
            
            # Add the content anyway and move to next section
            generated_sections = state["generated_sections"].copy()
            generated_sections.append({
                "title": f"{section_num}. {french_title}",
                "content": content + "\n\n_Note: This section reached maximum regeneration attempts and may require manual review._"
            })
            
            next_index = current_index + 1
            
            return {
                **state,
                "generated_sections": generated_sections,
                "is_approved": True,  # Force approval to move on
                "section_feedback": f"Maximum regeneration attempts reached for section: {french_title}. Using current content and moving to next section.",
                "current_section_index": next_index,
                "regeneration_attempts": 0
            }
            
        # If not approved, provide feedback for regeneration and increment counter
        print({**state,
            "regeneration_attempts": regeneration_attempts
        }["regeneration_attempts"])
        return {
            **state,
            "is_approved": False,
            # "section_feedback": review_result,
            "section_feedback":None,
            "regeneration_attempts": regeneration_attempts
        }
def finalize_document_node(state: DocumentState) -> DocumentState:
    """Generate the final PDF document with table of contents and color styling."""
    print("finalizer : ", state.get("regeneration_attempt",None))
    template = state["template"]
    generated_sections = state["generated_sections"]
    
    print(f"Finalizing document with {len(generated_sections)} approved sections")
    
    # Create in-memory PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add enhanced custom styles with colors
    styles.add(ParagraphStyle(
        'FrenchTitle', 
        parent=styles['Title'], 
        fontSize=20,
        spaceAfter=12,
        fontName="Helvetica-Bold",
        textColor=colors.blue,  # Blue text for main title
    ))
    
    styles.add(ParagraphStyle(
        'EnglishTitle', 
        parent=styles['Title'], 
        fontSize=16,
        spaceAfter=24,
        fontName="Helvetica-Bold",
        textColor=colors.navy,  # Navy text for secondary title
    ))
    
    # Create custom heading styles with colors
    styles.add(ParagraphStyle(
        'BlueHeading1',
        parent=styles['Heading1'],
        textColor=colors.blue,
        fontSize=16,
        spaceAfter=10,
    ))
    
    styles.add(ParagraphStyle(
        'BlueHeading2',
        parent=styles['Heading2'],
        textColor=colors.royalblue,
        fontSize=14,
        spaceAfter=8,
    ))
    
    styles.add(ParagraphStyle(
        'BlueHeading3',
        parent=styles['Heading3'],
        textColor=colors.steelblue,
        fontSize=12,
        spaceAfter=6,
    ))
    
    # Add cover page
    if generated_sections and generated_sections[0]["title"] == "Cover Page":
        doc_details = template["document_details"]
        
        elements.append(Paragraph(doc_details["document_name"], styles['FrenchTitle']))
        elements.append(Paragraph(doc_details["english_name"], styles['EnglishTitle']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Create document info table with improved styling
        data = [
            ["Document ID:", doc_details["document_id"]],
            ["Version:", doc_details["version"]],
            ["Date:", doc_details["date"]],
            ["Author:", doc_details["author"]],
            ["Approved by:", doc_details["approver"]]
        ]
        
        info_table = Table(data, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.steelblue),  # Blue background
            ('BACKGROUND', (1, 0), (1, -1), colors.white),  # White for content cells
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),  # White text on blue
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),  # Black text on white
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGNMENT', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(info_table)
        elements.append(Spacer(1, 1*inch))
        
        # Add purpose with blue heading
        elements.append(Paragraph("Purpose", styles['BlueHeading2']))
        elements.append(Paragraph(doc_details["purpose"]["description"], styles['Normal']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Add page break after cover
        elements.append(PageBreak())
    
    # Add table of contents if it exists
    if len(generated_sections) > 1 and generated_sections[1]["title"] == "Table of Contents":
        elements.append(Paragraph("Table of Contents", styles['BlueHeading1']))
        
        # Add TOC entries from policy sections with alternating backgrounds
        for i, section in enumerate(template["policy_sections"]):
            section_num = section.get("section_number", "")
            french_title = section.get("french_title", "")
            
            toc_entry = f"{section_num}. {french_title}"
            
            # Create a custom style for TOC entries with alternating backgrounds
            toc_style = ParagraphStyle(
                f'TOCEntry{i}',
                parent=styles['Normal'],
                backColor=colors.lightgrey if i % 2 == 0 else colors.white,
                spaceBefore=3,
                spaceAfter=3,
                leftIndent=6,
                rightIndent=6,
            )
            
            elements.append(Paragraph(toc_entry, toc_style))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(PageBreak())
    
    # Add content sections (skip cover page and TOC)
    for section_idx, section in enumerate(generated_sections[2:]):
        # Process Markdown content to ReportLab elements
        content = section["content"]
        
        # Add a section separator
        if section_idx > 0:
            # Create a gray separator line
            elements.append(Spacer(1, 0.2*inch))
            separator = Paragraph('<hr width="100%" color="lightgrey"/>', 
                                  ParagraphStyle('Separator', parent=styles['Normal']))
            elements.append(separator)
            elements.append(Spacer(1, 0.2*inch))
        
        # Process Markdown content line by line
        for line in content.split('\n'):
            # Handle headings with colors
            if line.strip().startswith('# '):
                heading_text = line.strip()[2:]
                elements.append(Paragraph(heading_text, styles['BlueHeading1']))
            elif line.strip().startswith('## '):
                heading_text = line.strip()[3:]
                elements.append(Paragraph(heading_text, styles['BlueHeading2']))
            elif line.strip().startswith('### '):
                heading_text = line.strip()[4:]
                elements.append(Paragraph(heading_text, styles['BlueHeading3']))
            # Handle bullet points with styling
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                bullet_text = line.strip()[2:]
                bullet_style = ParagraphStyle(
                    'Bullet',
                    parent=styles['Normal'],
                    leftIndent=20,
                    firstLineIndent=-15,
                )
                elements.append(Paragraph(f"• {bullet_text}", bullet_style))
            # Handle normal paragraphs
            elif line.strip():
                elements.append(Paragraph(line.strip(), styles['Normal']))
        
        # Add space between sections
        elements.append(Spacer(1, 0.3*inch))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    print("PDF document finalized successfully")
    
    return {
        **state,
        "complete": True,
        "pdf_data": buffer.getvalue()
    }
# Add nodes to the document generation graph
document_graph.add_node("document_generator", document_generator_node)
document_graph.add_node("document_reviewer", document_reviewer_node)
document_graph.add_node("finalize_document", finalize_document_node)

# Define graph flow with direct edges - simpler approach without conditional edges
document_graph.add_edge(START, "document_generator")
document_graph.add_edge("document_generator", "document_reviewer")

# Direct edges from reviewer based on state flags
# If not approved → back to generator
# If complete → to finalizer
# Otherwise (when approved) → back to generator for next section
document_graph.add_conditional_edges(
    "document_reviewer",
    lambda state: 
        "finalize_document" if state["complete"] else 
        "same_section_generator" if not state["is_approved"] else
        "next_section_generator",
    {
        "same_section_generator": "document_generator",
        "next_section_generator": "document_generator",
        "finalize_document": "finalize_document"
    }
)

document_graph.add_edge("finalize_document", END)

# Compile the document generation graph
compiled_document_graph = document_graph.compile()

def create_document_from_template(template_json):
    """Initialize and run document generation process."""
    print("Starting document generation process...")
    
    # Initial state
    state = {
        "template": template_json,
        "current_section_index": -1,  # Start with cover page (-1), then TOC (0), then sections (1+)
        "generated_sections": [],
        "section_content": "",
        "section_feedback": "",
        "is_approved": True,
        "complete": False,
        "pdf_data": None,
        "current_documents": []
        
    }
    
    # Run the graph with higher recursion limit to handle multiple sections
    result = compiled_document_graph.invoke(state, {"recursion_limit": 100})
    
    print(f"Document generation completed with {len(result['generated_sections'])} sections")
    return result

def load_json_template(file_path="AIMS.json"):
    """Load a JSON template file from the specified path."""
    try:
        # Use absolute path if provided, otherwise look in current directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)
            
        with open(file_path, 'r', encoding='utf-8') as file:
          return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in template file: {file_path}")
       