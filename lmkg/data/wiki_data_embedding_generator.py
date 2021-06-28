import pandas as pd
import spacy
import numpy as np

wiki_data_relation = pd.read_csv("wiki_data_relations.csv")
nlp = spacy.load("en_core_web_md")

relations_dict = {}
keywords_list = []
embeddings_list = []

for i, row in wiki_data_relation.iterrows():
    keywords = row["keywords"].split("\n")
    for keyword in keywords:
        doc = nlp(keyword)
        sent_list = [
            token.lemma_
            for sent in doc.sents
            for token in sent
            if token.text not in ["a", "an", "the"]
        ]
        sent = " ".join(sent_list)
        doc = nlp(sent)
        keywords_list.append(doc.text)
        embeddings_list.append(doc.vector)
        if keyword in relations_dict:
            relations_dict[doc.text].append(row["relation"])
        else:
            relations_dict[doc.text] = [row["relation"]]

keywords_list = np.array(keywords_list)
embeddings_list = np.array(embeddings_list)


np.save("wiki_data_relation_embeddings_md.npy", embeddings_list)
np.save("keywords_list_md.npy", keywords_list)
np.save("relations_dict_md.npy", relations_dict)
