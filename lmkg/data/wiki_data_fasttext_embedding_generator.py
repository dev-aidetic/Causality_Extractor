import pandas as pd
import spacy
import numpy as np

import sys
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_PATH)

from utils import get_fasttext_vectors

fasttext_model = get_fasttext_vectors.FastText()

wiki_data_relation = pd.read_csv("relationship_analysis.csv")
wiki_data_relation.dropna(inplace=True)
wiki_data_relation = wiki_data_relation.loc[wiki_data_relation.Use == 1].copy()

nlp = spacy.load("en_core_web_md")
relations_dict = {}
keywords_list = []
embeddings_list = []
for i, row in wiki_data_relation.iterrows():
    try:
        keywords = row["aliases"].split(", ")
        keywords = keywords + [row["label"]]
        keywords = list(set(keywords))
    except Exception as e:
        print(e)
        continue
    generated_keywords = []
    for keyword in keywords:
        doc = nlp(keyword)
        token_list = []
        for token in doc:
            if token.text not in ["a", "an", "the"]:
                token_list.append(token.lemma_)
        keyword = " ".join(token_list)
        if keyword in generated_keywords:
            continue
        try:
            generated_keywords.append(keyword)
            doc = nlp(keyword)
            embeddings_list.append(fasttext_model.get_sent_vectors(keyword))
            keywords_list.append(doc.text)
            if keyword in relations_dict:
                relations_dict[doc.text].append(row["label"])
            else:
                relations_dict[doc.text] = [row["label"]]
        except Exception as e:
            print(e)
keywords_list = np.array(keywords_list)
embeddings_list = np.array(embeddings_list)

np.save("wiki_data_relation_embeddings_ft.npy", embeddings_list)
np.save("keywords_list_ft.npy", keywords_list)
np.save("relations_dict_ft.npy", relations_dict)
