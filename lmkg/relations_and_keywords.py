import pandas as pd
import spacy
import numpy as np

wiki_data_relation = pd.read_csv("data/cause-effect-keywords.csv")
nlp = spacy.load("en_core_web_md")

relations_dict = {}
keywords_list = []
embeddings_list = []
for i, row in wiki_data_relation.iterrows():
    try:
        keywords = row["aliases"].split(", ")
    except:
        continue
    for keyword in keywords:
        doc = nlp(keyword)
        token_list = []
        for token in doc:
            if token.text not in ["a", "an", "the"]:
                token_list.append(token.lemma_)
        keyword = " ".join(token_list)
        doc = nlp(keyword)
        keywords_list.append(doc.text)
        embeddings_list.append(doc.vector)
        if keyword in relations_dict:
            relations_dict[doc.text].append(row["label"])
        else:
            relations_dict[doc.text] = [row["label"]]

keywords_list = np.array(keywords_list)
embeddings_list = np.array(embeddings_list)

np.save("data/cause_effect_wiki_data_relation_embeddings_md.npy", embeddings_list)
np.save("data/cause_effect_keywords_list_md.npy", keywords_list)
np.save("data/cause_effect_relations_dict_md.npy", relations_dict)
