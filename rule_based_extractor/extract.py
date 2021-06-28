import spacy
import re
import nltk
from .extract_helper import RuleBasedExtractor
import itertools
import contractions

mapper = {
    "caused by": 0,
    "driven by": 0,
    "impacted by": 0,
    "informed by": 0,
    "amplified by": 0,
    "affected by": 0,
    "offset by": 1,
    "resulting in": 1,
    "results in": 1,
    "result in": 1,
    "resulted in": 1,
    "causing in": 1,
    "led to": 1,
    "leads to": 1,
    "leading to": 1,
}


def clean_text(raw_text):
    expanded_words = []
    for word in raw_text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = " ".join(expanded_words)
    clean_text = re.sub(r"\([^()]*\)", " ", expanded_text)
    clean_text = re.sub(r"[?|!]", ". ", clean_text)
    clean_text = re.sub(r"[^a-zA-Z0-9. ]", " ", clean_text)
    clean_text = re.sub(r"\n", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text


class GetCausalReln:
    def __init__(
        self, nlp, txt_file, np_link=True, load_from_file=False, controlled=True
    ):
        self.nlp = nlp
        self.txt_file = txt_file
        self.load_from_file = load_from_file
        self.controlled = controlled
        self.np_link = np_link

    def uncontrolled_extraction(self):
        sentences = []
        for path in self.txt_file:
            with open(path, "r") as f:

                for line in f:
                    lines = line.strip()
                    lines = self.nlp(lines).sents
                    for line in lines:
                        sents = self.causal_sentence_finder(line.text)
                        if sents:
                            sentences.extend(sents)
        print(f"found {len(sentences)} sentences")
        return sentences

    def determine_structure(self, text, comp):

        doc = self.nlp(text)
        for token in doc:
            if token.dep_ == "ROOT":
                root = token.text
                break

        for token in doc:
            if token.dep_ in ["agent", "prep"] and token.head.text == root:
                if token.text == comp:

                    return {"sentence": text, "root": root, "comp": comp}
        return {}

    def causal_sentence_finder(self, text):

        sent_text = nltk.sent_tokenize(text)
        sentences = []

        for sentence in sent_text:

            if re.search(rf"\bin\b", sentence):
                txt = self.determine_structure(text, "in")
                if txt:
                    sentences.append(txt)

            elif re.search(rf"\bby\b", sentence):
                txt = self.determine_structure(text, "by")
                if txt:
                    sentences.append(txt)
            elif re.search(rf"\bto\b", sentence):
                txt = self.determine_structure(text, "by")
                if txt:
                    sentences.append(txt)

        return sentences

    def identify_causal_sentences(self, text):

        sent_text = nltk.sent_tokenize(text)
        sentences = []

        for sentence in sent_text:
            for keyword in mapper:
                if re.search(rf"\b{keyword}\b", sentence):
                    comp = "by"
                    if re.search(r" by", keyword):
                        comp = "by"
                    elif re.search(r" in", keyword):
                        comp = "in"
                    elif re.search(r" to", keyword):
                        comp = "to"
                    root = re.sub(rf" {comp}", "", keyword)
                    sentences.append({"sentence": sentence, "root": root, "comp": comp})
        sentences = [sentence_ for sentence_ in sentences if sentence_]
        return sentences

    def extract_sentences(self):
        sentences = []
        for path in self.txt_file:
            with open(path, "r") as f:

                for line in f:
                    lines = line.strip()
                    lines = lines.lower()
                    lines = self.nlp(lines).sents
                    for line in lines:
                        sents = self.identify_causal_sentences(line.text)
                        if sents:
                            sentences.extend(sents)
        print(f"found {len(sentences)} sentences")
        return sentences

    def check_sentence(self, key, sentence):
        regex = re.compile(rf"\b{key}\b", re.I)
        return regex.findall(sentence) != []

    def get_causal_relations(self):
        relation_dict = []
        if self.load_from_file:
            if self.controlled:
                sentences = self.extract_sentences()
            else:
                sentences = self.uncontrolled_extraction()
        else:
            sentences = []
            if self.controlled:
                sentences = self.identify_causal_sentences(self.txt_file)
            else:
                sentences = self.causal_sentence_finder(self.txt_file)
        triplet_extractor = RuleBasedExtractor(self.nlp, self.np_link)

        print("Total Number of input sentences :", len(sentences))
        for sents in sentences:
            try:

                if mapper[sents["root"] + f" {sents['comp']}"] == 0:
                    cause, effect = triplet_extractor.primary_triplet_extraction(
                        sents["sentence"], sents["root"], sents["comp"]
                    )

                elif mapper[sents["root"] + f" {sents['comp']}"] == 1:
                    effect, cause = triplet_extractor.primary_triplet_extraction(
                        sents["sentence"], sents["root"], sents["comp"]
                    )
                if cause and effect:
                    relation_dict.append(
                        {
                            "sentence": clean_text(sents["sentence"]),
                            "cause": cause,
                            "effect": effect,
                            "relation": sents["root"],
                        }
                    )
            except Exception as e:
                print(e)
                print(sents)

        print(f"Total relations extracted is {len(relation_dict)}")

        return relation_dict
