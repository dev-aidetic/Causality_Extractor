import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

relations_dict = np.load(
    "data/cause_effect_relations_dict_md.npy", allow_pickle=True
).item()
keyword_embedding = np.load(
    "data/cause_effect_wiki_data_relation_embeddings_md.npy", allow_pickle=True
)
keyword_name = np.load("data/cause_effect_keywords_list_md.npy", allow_pickle=True)


def MapHasCauseRelations(lmkg_triplets,nlp):
    for triplet in lmkg_triplets:
        if triplet["c"] < 0.03:
            continue
        lm_relation = triplet["r"]
        lm_relation = nlp(" ".join(lm_relation))
        lm_relation_emb = lm_relation.vector
        cosine_sims = cosine_similarity(
            lm_relation_emb.reshape(1, -1), keyword_embedding
        )
        cosine_sims = cosine_sims.reshape(
            cosine_sims.shape[1],
        )
        max_idx = np.argmax(cosine_sims)
        if cosine_sims[max_idx] > 0.85:
            triplet["mapped_relations"] = {
                "wiki_data_relation": [*relations_dict[keyword_name[max_idx]]],
                "similarity_score": float(cosine_sims[max_idx]),
            }
        else:
            triplet["mapped_relations"] = {
                "wiki_data_relation": [],
                "similarity_score": 0,
            }

    return lmkg_triplets


if __name__ == "__main__":
    pass
