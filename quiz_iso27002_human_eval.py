import streamlit as st
import json
from typing import List, Tuple
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
# Quiz generator with human eval
# -----------------------------
def generate_quiz_question_1(section, subsection, text):
    """ 
    Use LLM to generate a variable number of quiz questions for a specific ISO clause 
    """
    # Get the reference ID from the subsection (first part before space) 
    ref_parts = subsection.split(" ", 1) 
    ref = ref_parts[0] if len(ref_parts) > 0 else "unknown"
    
    # Create a more focused prompt for the LLM 
    prompt = f""" 
Agissez en tant qu’auditeur principal certifié ISO 27002:2022.

Vous êtes en train d’examiner la clause {subsection} : « {text} ».

Votre mission est de générer une série réaliste de questions de type audit, suivant une progression logique et en chaîne. Ces questions doivent simuler la manière dont un auditeur expérimenté approfondirait une évaluation de conformité réelle.

Pour chaque question, fournissez immédiatement après une réponse plausible et pertinente que l’organisation auditée pourrait donner, en lien avec le contexte décrit. Les réponses doivent être réalistes, refléter les bonnes pratiques, et s’aligner avec les exigences de la clause.

Contexte :
L’organisation auditée est une entreprise spécialisée dans les logiciels et les technologies. Elle comprend des équipes de développement logiciel ainsi que des équipes de validation. Ces équipes interagissent fréquemment avec le client télécom (opérateur) en Asie et en Europe, et collaborent avec des partenaires tiers. Les activités de développement, la validation du code, l’intégration et le déploiement continus, ainsi que la collaboration externe font partie intégrante de leur environnement opérationnel.

Consignes :
- Commencez par une question générale pour évaluer la conformité de base.
- Ensuite, posez des questions de plus en plus détaillées ou approfondies, en fonction des réponses probables.
- Les questions doivent être spécifiques, exploitables et clairement liées au contenu de la clause.
- Fournissez une réponse réaliste juste après chaque question.
- Adoptez un ton naturel et humain, tel qu’un auditeur le ferait lors d’un entretien.
- Incluez 1 et un seul seulement  question/réponse.
- Assurez-vous que chaque question s’appuie sur la précédente — comme dans une conversation réfléchie ou une visite d’audit.
- Formulez vos questions et réponses en tenant compte de l’environnement de l’entreprise : développement logiciel, validation, interactions clients et collaborations avec des tiers.
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
        qna_pairs.append({
            "question": question,
            "answer": answer
        })

    return {
        "section": section,
        "subsection": subsection,
        "ref_id": ref,
        "qna1": qna_pairs,
        "text": text  # Include the clause content
    }





# Define gemma_model at the top of your script


def evaluate_user_response_1(user_response: str, model_answer: str, iso_text: str) -> Tuple[int, str]:
    """
    Evaluate user response using Gemma model directly.
    """
    if not user_response or user_response.strip() == "":
        return 0, "No response provided"
    
    # Create the prompt for evaluation
    prompt = f"""
Agissez en tant qu'auditeur principal certifié ISO 27002:2022.

Vous devez évaluer la réponse d'un candidat auditeur à une question d'audit par rapport à la norme ISO et à la réponse modèle.

Contexte de la norme ISO:
{iso_text}

Réponse modèle:
{model_answer}

Réponse du candidat:
{user_response}

Votre tâche:
1. Évaluez la réponse du candidat par rapport à la réponse modèle et au contexte de la norme ISO.
2. Déterminez un pourcentage de conformité (0% à 100%) qui représente à quel point la réponse du candidat est alignée avec les exigences de la norme ISO.
3. Fournissez un commentaire constructif qui explique ce qui est correct et ce qui pourrait être amélioré.

Format de votre réponse:
- Pourcentage de conformité: X%
- Commentaire: [Votre commentaire d'évaluation]

Ne donnez AUCUNE introduction ou conclusion - fournissez uniquement les deux éléments demandés dans le format exact spécifié ci-dessus.
"""
    
    try:
        response = gemma_model.invoke(prompt).content
        
        lines = response.strip().split("\n")
        percentage_line = next((line for line in lines if "Pourcentage" in line or "pourcentage" in line.lower()), "")
        percentage_str = percentage_line.split(":")[-1].strip().replace("%", "")
        
        try:
            percentage = int(percentage_str) if percentage_str.isdigit() else 75
        except:
            percentage = 75
        
        feedback_index = next((i for i, line in enumerate(lines) if "Commentaire" in line or "commentaire" in line.lower()), -1)
        if feedback_index >= 0 and feedback_index < len(lines) - 1:
            feedback = "\n".join(lines[feedback_index+1:])
        else:
            feedback = next((line.split(":", 1)[1].strip() for line in lines if "Commentaire" in line or "commentaire" in line.lower()), 
                          "Veuillez comparer votre réponse avec la réponse suggérée.")
            
        return percentage, feedback
        
    except Exception as e:
        print(f"LLM evaluation error: {str(e)}")
        return 70, "L'évaluation automatique n'a pas pu être complétée. Veuillez comparer votre réponse avec la réponse suggérée."

def display_quiz_in_streamlit_1(quiz_results):
    """Display the quiz questions and collect responses only without evaluation"""
    st.markdown("## ISO 27002:2022 Audit Questions")

    # Initialize session state for storing responses if not already present
    if "quiz_responses" not in st.session_state:
        st.session_state.quiz_responses = {}

    tabs = st.tabs([item["subsection"] for item in quiz_results])

    for i, (tab, item) in enumerate(zip(tabs, quiz_results)):
        subsection = item["subsection"]
        ref_id = item["ref_id"]
        iso_text = item["text"]
        qna = item["qna"]

        with tab:
            st.markdown(f"**Subsection:** {subsection} ({ref_id})")

            # Create a form for each tab to prevent premature submission
            with st.form(key=f"form_{ref_id}"):
                tab_responses = {}
                
                for q_idx, qa in enumerate(qna):
                    question = qa["question"]
                    model_answer = qa["answer"]
                    response_key = f"{ref_id}_{q_idx}"

                    st.markdown(f"**Question #{q_idx + 1}:** *{question}*")

                    with st.expander("Show suggested answer"):
                        st.markdown(f"*{model_answer}*")

                    # Pre-fill with existing response if available
                    initial_value = ""
                    if response_key in st.session_state.quiz_responses:
                        initial_value = st.session_state.quiz_responses[response_key]["user_response"]

                    # Use a text area that doesn't submit on Enter
                    user_response = st.text_area(
                        f"Your response to question #{q_idx + 1}:",
                        value=initial_value,
                        key=f"input_{response_key}",
                        height=150  # Add some height for more comfortable typing
                    )

                    # Store response metadata for this question
                    tab_responses[response_key] = {
                        "question": question,
                        "model_answer": model_answer,
                        "user_response": user_response,
                        "iso_text": iso_text,
                        "ref_id": ref_id,
                        "q_idx": q_idx
                    }
                
                # Add a submit button for this tab's form
                submitted = st.form_submit_button("Save Responses for This Section")
                if submitted:
                    # Update session state with this tab's responses
                    for key, value in tab_responses.items():
                        st.session_state.quiz_responses[key] = value
                    st.success(f"Responses for section {subsection} saved successfully!")
    
    # Add clear button with a unique key - note we removed the evaluate button
    if st.button("🗑️ Clear All Responses", key="clear_btn_display"):
        st.session_state.quiz_responses = {}
        if "evaluation_results" in st.session_state:
            del st.session_state.evaluation_results
        st.success("All responses cleared!")
        

def display_evaluation_results_1(evaluation_results):
    """Display evaluation results in a separate section"""
    st.markdown("---")
    st.markdown("## 📊 Evaluation Results")
    
    # Calculate overall score
    if evaluation_results:
        total_score = sum(result["percentage"] for result in evaluation_results)
        avg_score = total_score / len(evaluation_results)
        st.markdown(f"### Overall Score: {avg_score:.1f}%")
    
    # Group results by ref_id for better organization
    grouped_results = {}
    for result in evaluation_results:
        key_parts = result["key"].split("_")
        ref_id = key_parts[0] if len(key_parts) > 0 else "unknown"
        
        if ref_id not in grouped_results:
            grouped_results[ref_id] = []
        
        grouped_results[ref_id].append(result)
    
    # Display results grouped by section
    for ref_id, results in grouped_results.items():
        with st.expander(f"Section {ref_id} - {len(results)} questions", expanded=True):
            for result in results:
                col1, col2 = st.columns([1, 5])
                with col1:
                    # Display score with color coding
                    score = result["percentage"]
                    if score >= 80:
                        st.markdown(f"<h3 style='color:green'>{score}%</h3>", unsafe_allow_html=True)
                    elif score >= 60:
                        st.markdown(f"<h3 style='color:orange'>{score}%</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color:red'>{score}%</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Q: {result['question']}**")
                    st.markdown(f"*Your answer:* {result['user_response']}")
                    st.markdown(f"*Feedback:* {result['feedback']}")
                
                st.markdown("---")
    
    # Option to download results as CSV
    if evaluation_results:
        import pandas as pd
        import io
        
        # Convert to dataframe for CSV export
        df = pd.DataFrame([
            {
                "Question": r["question"],
                "Your Response": r["user_response"],
                "Score": r["percentage"],
                "Feedback": r["feedback"]
            } for r in evaluation_results
        ])
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Evaluation Results",
            data=csv,
            file_name="iso27002_quiz_results.csv",
            mime="text/csv",
        )
def evaluate_quiz_responses_1():
    """
    Evaluate all quiz responses using the LLM and return results
    """
    if not st.session_state.quiz_responses:
        st.warning("No responses to evaluate. Please fill in at least one response first.")
        return None
        
    # Display a progress indicator
    with st.spinner("Evaluating your responses..."):
        # Create a copy to avoid modifying during iteration
        responses_to_evaluate = dict(st.session_state.quiz_responses)
        evaluation_results = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        total_items = len(responses_to_evaluate)
        
        for i, (key, entry) in enumerate(responses_to_evaluate.items()):
            # Update progress
            progress_bar.progress((i + 1) / total_items)
            
            user_response = entry["user_response"].strip()
            if user_response:
                percentage, feedback = evaluate_user_response_1(
                    user_response,
                    entry["model_answer"],
                    entry["iso_text"]
                )
                evaluation_results.append({
                    "key": key,
                    "question": entry["question"],
                    "model_answer": entry["model_answer"],
                    "user_response": user_response,
                    "percentage": percentage,
                    "feedback": feedback
                })
            else:
                evaluation_results.append({
                    "key": key,
                    "question": entry["question"],
                    "model_answer": entry["model_answer"],
                    "user_response": "",
                    "percentage": 0,
                    "feedback": "Aucune réponse saisie."
                })
    
    # Return the evaluation results
    return evaluation_results            
def generate_iso_quiz_1():
    print("🔍 Generating ISO 27002 quiz questions...")

    # Load JSON file
    with open("new.json", 'r', encoding='utf-8') as f:
        structured_sections = json.load(f)

    quiz_results = []

    # Iterate through each category in the JSON structure
    for category, items in structured_sections.items():
        print(f"\n📂 Category: {category}")

        for entry in items:
            try:
                # Split the entry by comma to get subsection and section
                subsection, section = map(str.strip, entry.split(",", 1))
            except ValueError:
                print(f"❌ Skipping malformed entry: {entry}")
                continue

            query = f"{subsection} {section}"
            print(f"\n🔹 Searching for: {query}")

            results = vector_store_i_title.similarity_search(query=query, k=1)

            if results:
                combined_context = ""
                for doc in results:
                    title = doc.metadata.get("title")
                    content = doc.metadata.get("content", "")
                    print(f"✅ Found title: {title}")
                    print(f"📄 Content snippet: {content[:200]}...\n")

                    combined_context += content + "\n"

                if combined_context.strip():
                    quiz_item = generate_quiz_question_1(section, subsection, combined_context.strip())
                    quiz_item["category"] = category  # Include the category
                    quiz_results.append(quiz_item)
                else:
                    print(f"❌ No valid content for: {subsection}")
            else:
                print(f"❌ No matches found for: {query}")

    return quiz_results