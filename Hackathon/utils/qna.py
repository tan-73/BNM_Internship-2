# utils/qna.py
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")


def summarize_document(chunks):
    text = "\n\n".join([chunk.page_content for chunk in chunks])
    text_chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = [summarizer(chunk, max_length=256, min_length=60, do_sample=False)[0]['summary_text']
                 for chunk in text_chunks]
    return " ".join(summaries)


def answer_question_with_rag(question, vectorstore, k=3):
    relevant_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])[:2000]  # cap for token limit
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = qa_model(prompt, max_length=256, do_sample=False)
    return response[0]['generated_text']
