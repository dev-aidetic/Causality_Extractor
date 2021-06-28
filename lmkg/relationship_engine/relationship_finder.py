import spacy
import numpy as np
import re
from typing import List
from transformers import AutoTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_PATH, "data")

RELATIONSHIP_THRESOLD = 0.8

from kg_builder import process, mapper, stop_words
from utils import utils  # , get_fasttext_vectors

logger = utils.intialize_logging(__name__)

# fasttext_model = get_fasttext_vectors.FastText()


class RelationshipFinder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.language_model = "bert-large-cased"
        self.use_cuda = False
        self.threshold = 0.02
        try:
            self.language_model = os.path.join(MODEL_PATH, self.language_model)
            logger.info(f"Model path: {self.language_model}")
            logger.info(os.listdir(self.language_model))
            self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)
            self.encoder = BertModel.from_pretrained(self.language_model)
        except Exception as e:
            logger.info("Model is not present in data folder, donwloading from source")
            logger.info(e)
            self.language_model = "bert-large-cased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)
            self.encoder = BertModel.from_pretrained(self.language_model)
        logger.info(f"Model path: {self.language_model}")
        self.encoder.eval()
        if self.use_cuda:
            self.encoder = self.encoder.cuda()

        if os.path.exists("data/wiki_data_relation_embeddings_cause_md.npy"):
            self.keyword_embeddings = np.load(
                "data/wiki_data_relation_embeddings_cause_md.npy", allow_pickle=True
            )
            print(self.keyword_embeddings.shape)
        else:
            raise Exception("wikidata keyword embeddings is not present in data folder")

        if os.path.exists("data/relations_dict_cause_md.npy"):
            self.relations_dict = np.load(
                "data/relations_dict_cause_md.npy", allow_pickle=True
            ).item()
            print(len(self.relations_dict))
        else:
            raise Exception("keyword to relation mapping is not present in data folder")

        if os.path.exists("data/keywords_list_cause_md.npy"):
            self.keyword_list = np.load(
                "data/keywords_list_cause_md.npy", allow_pickle=True
            )
            print(self.keyword_list)
        else:
            raise Exception("wikidata keyword list is not present in data folder")

    def find_from_text(self, text: str):
        """Find relationships from text"""
        text_proc = text.replace("\n", ".").strip()
        if text_proc:
            output_triplets = {"text": text, "relations": []}
            nlp_text = self.nlp(text_proc)
            for sent in nlp_text.sents:
                proc_triplets = []
                adv_list = [token.lemma_ for token in sent if token.pos_ == "ADV"]
                stop_word_list = stop_words.stop_words_list + adv_list
                for triplet in process.parse_sentence(
                    sent.text,
                    self.tokenizer,
                    self.encoder,
                    self.nlp,
                    use_cuda=self.use_cuda,
                ):
                    head = utils.clean_text(triplet["h"], stop_word_list)
                    tail = utils.clean_text(triplet["t"], stop_word_list)
                    relations = triplet["r"]
                    conf = triplet["c"]
                    if conf >= self.threshold:
                        proc_triplet = mapper.Map(head, relations, tail)
                        if "h" in proc_triplet:
                            proc_triplet["c"] = conf
                            proc_triplets.append(proc_triplet)

                de_dup_triplets = mapper.deduplication(proc_triplets)
                for de_dup_triplet in de_dup_triplets:
                    (
                        relationship_label,
                        relationship_score,
                    ) = self.map_with_existing_relations(de_dup_triplet)
                    de_dup_triplet["wikidata_relation"] = relationship_label
                    de_dup_triplet["wikidata_relation_score"] = round(
                        relationship_score, 2
                    )
                    de_dup_triplet["sent"] = sent.text

                output_triplets["relations"].extend(de_dup_triplets)

                output_triplets["relations"] = sorted(
                    output_triplets["relations"],
                    key=lambda x: (x["wikidata_relation_score"], x["c"]),
                    reverse=True,
                )
                output_triplets["relations"] = [
                    i
                    for i in output_triplets["relations"]
                    if (i["wikidata_relation"] != "NO_MATCH")
                    and (" ".join(i["r"]) not in stop_words.stop_words_list)
                ]

            return output_triplets
        else:
            return None

    def map_with_existing_relations(self, triplet: dict):
        """Map the relationship with wikidata relationships

        Args:
            triplet (dict): relationship triplet
        """
        lm_relation = triplet["r"]
        lm_relation = " ".join(lm_relation)
        lm_relation = self.nlp(lm_relation)
        try:
            # lm_relation_emb = fasttext_model.get_sent_vectors(lm_relation)
            lm_relation_emb = lm_relation.vector
            cosine_sims = cosine_similarity(
                lm_relation_emb.reshape(1, -1), self.keyword_embeddings
            )
            cosine_sims = cosine_sims.reshape(
                cosine_sims.shape[1],
            )
            max_idx = np.argmax(cosine_sims)
            if cosine_sims[max_idx] >= RELATIONSHIP_THRESOLD:
                relationship_label = self.relations_dict[self.keyword_list[max_idx]]
                relationship_score = float(cosine_sims[max_idx])
                return relationship_label, relationship_score
        except Exception as e:
            logger.info("can not generate fasttext embedding")
            logger.info(e)
            return "NO_MATCH", 0
        else:
            return "NO_MATCH", 0