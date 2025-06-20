# utils/clause_checker.py
import torch

CLAUSES = {
    "identity_of_controller": ["controller", "data controller", "identity"],
    "lawful_basis": ["lawful basis", "consent", "contract", "legitimate interest"],
    "data_subject_rights": ["access", "erasure", "portability", "rectification", "withdraw"],
    "data_retention": ["retention", "stored", "retain"],
    "third_party_transfers": ["third party", "data sharing", "recipient"],
    "international_transfers": ["adequacy decision", "standard contractual clause"],
    "security_measures": ["encryption", "security", "access control"],
    "dpo_contact": ["data protection officer", "DPO", "contact"],
    "breach_notification": ["data breach", "breach notification", "72 hours"]
}

def detect_clauses(chunks, embedder):
    texts = [chunk.page_content for chunk in chunks]
    doc_embeddings = embedder.embed_documents(texts)
    clause_presence = {}

    for clause_name, keywords in CLAUSES.items():
        query = " ".join(keywords)
        query_emb = embedder.embed_query(query)
        sims = [torch.nn.functional.cosine_similarity(
                    torch.tensor(query_emb), torch.tensor(doc_emb), dim=0).item()
                for doc_emb in doc_embeddings]
        clause_presence[clause_name] = max(sims) > 0.55

    return clause_presence

def clause_compliance_score(presence_dict):
    score = round((sum(presence_dict.values()) / len(presence_dict)) * 100, 2)
    return score
