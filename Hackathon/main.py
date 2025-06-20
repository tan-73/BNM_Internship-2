# main.py
import streamlit as st
from utils.document_loader import extract_text_from_pdf
from utils.clause_checker import detect_clauses, CLAUSES, clause_compliance_score
from utils.qna import answer_question_with_rag, summarize_document
from utils.embedding import load_embedder, load_vectorstore, split_into_chunks

st.set_page_config("GDPR Compliance Verifier", layout="wide")

st.title("🛡️ GDPR Compliance Verifier + AI Assistant")

uploaded = st.file_uploader("Upload a Privacy Policy or DPA (PDF)", type="pdf")

if uploaded:
    with st.spinner("Extracting text..."):
        docs = extract_text_from_pdf(uploaded)

    st.success(f"Extracted {len(docs)} pages")
    text_chunks = split_into_chunks(docs)
    embedder = load_embedder()
    vectorstore = load_vectorstore(text_chunks, embedder)

    with st.expander("🔍 Clause Detection"):
        presence = detect_clauses(text_chunks, embedder)
        score = clause_compliance_score(presence)
        st.metric("Estimated GDPR Compliance Score", f"{score}%")
        for k, v in presence.items():
            clause = k.replace("_", " ").title()
            st.write(f"{'✅' if v else '❌'} {clause}")

    with st.expander("🧠 Document Summary"):
        if st.button("Summarize Document"):
            summary = summarize_document(text_chunks)
            st.write(summary)

    st.subheader("💬 Ask Questions About the Document")
    q = st.text_input("Ask a legal/compliance question:",
                      placeholder="e.g., What is the retention period?")
    if q:
        with st.spinner("Searching for answer..."):
            answer = answer_question_with_rag(q, vectorstore)
            st.write(answer)

else:
    st.info("Please upload a document to begin.")
