
import streamlit as st

import json



from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import json
import pandas as pd
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
# Quiz generator iso 42001
# -----------------------------
def generate_quiz_question_2(section, subsection, text):
    """ 
    Use LLM to generate a variable number of quiz questions for a specific ISO clause 
    """
    # Get the reference ID from the subsection (first part before space) 
    ref_parts = subsection.split(" ", 1) 
    ref = ref_parts[0] if len(ref_parts) > 0 else "unknown"
    
    # Create a more focused prompt for the LLM 
    prompt = f""" 
Agissez en tant qu’auditeur principal certifié lead auditor ISO 27002:2022.

Vous êtes en train d’examiner la clause ISO {subsection} : « {text} ».

Votre mission est de générer une série réaliste de questions de type audit, suivant une progression logique et en chaîne. Ces questions doivent simuler la manière dont un auditeur expérimenté approfondirait une évaluation de conformité réelle.

Pour chaque question, fournissez immédiatement après une réponse plausible et pertinente que l’organisation auditée pourrait donner, en lien avec le contexte décrit. Les réponses doivent être réalistes, refléter les bonnes pratiques, et s’aligner avec les exigences de la clause.


Consignes :

- Incluez seulement 1 seul  paires question/réponse.


- N’INCLUEZ AUCUNE mise en forme, puce ou explication — produisez uniquement les questions suivies de leur réponse sur deux lignes consécutives.

Générez maintenant l’ensemble des paires questions/réponses en chaîne pour cette clause, en français. 

"""

    response = gemma_model.invoke(prompt).content

    # Split into lines, and group them as Q&A pairs
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    qna_pairs = []

    for i in range(0, len(lines) - 1, 2):
        question = lines[i]
        answer = lines[i + 1]
        #company_answer, sources = ask_company_question(question)
        qna_pairs.append({
            "question": question,
            "answer": answer,
            #"company_answer": company_answer,
            #"sources": sources
        })

    return {
        "section": section,
        "subsection": subsection,
        "ref_id": ref,
        "qna": qna_pairs,
        "text": text  # Include the clause content
    }

def save_quiz_to_csv_2(quiz_results):
    """
    Save quiz results directly (without flattening) to a CSV-compatible JSON format.
    """
    import os
    import json

    filename = "iso42001_quiz1_output_raw.json"  # Better suited for structured data
    file_path = os.path.abspath(filename)

    # Save directly as JSON for structured access later
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(quiz_results, f, ensure_ascii=False, indent=2)

    print(f"Raw quiz results saved to: {file_path}")
    return file_path

    
    

def display_quiz_in_streamlit_2(quiz_results):
    

    st.markdown("## ISO 27002:2022 Audit Questions")

    for i, item in enumerate(quiz_results):
        section = item["section"]
        subsection = item["subsection"]
        ref_id = item["ref_id"]
        qna = item["qna"]

        with st.expander(f"{subsection}", expanded=(i == 0)):
            st.markdown(f"**Section:** {section}")
            st.markdown(f"**Reference:** {ref_id}")

            questions_df = pd.DataFrame({
                "Question #": range(1, len(qna) + 1),
                "Audit Question": [q["question"] for q in qna],
                "Suggested Answer": [q["answer"] for q in qna],
                #"Company Answer": [q["company_answer"] for q in qna],
                #"Sources": [", ".join(q["sources"]) if q["sources"] else "—" for q in qna]
            })

            st.table(questions_df)

            section_csv = questions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download '{ref_id}' Questions",
                data=section_csv,
                file_name=f"iso42001_{ref_id}_questions.csv",
                mime="text/csv",
                key=f"download_section_{i}"
            )

    all_questions = []
    for item in quiz_results:
        for i, qna in enumerate(item["qna"]):
            all_questions.append({
                "Section": item["section"],
                "Subsection": item["subsection"],
                "Reference ID": item["ref_id"],
                "Question #": i + 1,
                "Question": qna["question"],
                "Answer": qna["answer"],
                #"Company Answer": qna["company_answer"],
                #"Sources": ", ".join(qna["sources"]) if qna["sources"] else "—"
            })

    all_questions_df = pd.DataFrame(all_questions)

    if all_questions:
        st.markdown("### Download All Questions")
        full_csv = all_questions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Question Set",
            data=full_csv,
            file_name="iso42001_all_questions.csv",
            mime="text/csv",
            key="download_all"
        )


import json

def generate_iso_quiz_2():
    print("🔍 Generating ISO 42001 quiz questions...")

    with open("new.json", 'r', encoding='utf-8') as f:
        structured_sections = json.load(f)

    quiz_results = []

    for category, items in structured_sections.items():
        print(f"\n📂 Category: {category}")

        for entry in items:
            try:
                subsection, section = map(str.strip, entry.split(",", 1))
            except ValueError:
                print(f"❌ Skipping malformed entry: {entry}")
                continue

            query = f"{subsection} {section}"
            print(f"\n🔹 Searching for: {query}")

            results = vector_store_iso_title24001.similarity_search(query=query, k=1)

            if results:
                combined_context = ""
                for doc in results:
                    title = doc.metadata.get("title", "")
                    content = doc.metadata.get("content", "")
                    source = doc.metadata.get("source", "")
                    print(f"✅ Found title: {title}")
                    print(f"📄 Content snippet: {content[:200]}...\n")

                    combined_context += f"{title}\n{content}\n{source}\n"

                if combined_context.strip():
                    quiz_item = generate_quiz_question_2(section, subsection, combined_context.strip())
                    quiz_item["category"] = category  # optionally include the category
                    quiz_results.append(quiz_item)
                else:
                    print(f"❌ No valid content for: {subsection}")
            else:
                print(f"❌ No matches found for: {query}")

    return quiz_results


