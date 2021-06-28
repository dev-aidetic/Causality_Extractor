from nltk.util import pr
import spacy
import re


class RuleBasedExtractor:
    def __init__(self, nlp, np_link=True):
        self.nlp = nlp
        self.np_link = np_link

    def prep_linking(self, prep_effect, prep_cause, doc, cause, effect):
        count_c = 0
        count_e = 0
        for chunk in doc.noun_chunks:

            if chunk.root.head == prep_effect:
                if count_e == 0:
                    effect = " ".join([effect, str(prep_effect), str(chunk)])
                else:
                    effect = " ".join([effect, "and", str(chunk)])

                count_e += 1

            if chunk.root.head == prep_cause:
                if count_c == 0:

                    cause = " ".join([cause, str(prep_cause), str(chunk)])
                else:
                    cause = " ".join([cause, "and", str(chunk)])

                count_c += 1

            return cause, effect

    def secondary_triplet_extraction(self, doc, ROOT_KEYWORD, comp, ROOT):

        cause = ""
        effect = ""
        prep_cause = None
        prep_effect = None
        np_effect = ""
        np_cause = ""
        np_preps = []
        np_list = []
        for token in doc:

            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.text == ROOT_KEYWORD:

                for child in token.children:
                    if child.dep_ == "prep":
                        prep_cause = child

        for chunk in doc.noun_chunks:
            np_list.append(chunk.text)
            if chunk.root.head.dep_ == "prep":
                np_preps.append(chunk.root.head)

        for chunk in doc.noun_chunks:

            if (
                chunk.root.dep_ in ["nsubj", "nsubjpass", "ROOT"]
                and chunk.root.head.text == ROOT
            ):

                effect = str(chunk)
                np_effect = str(chunk)

            if chunk.root.head.text == comp:
                cause = str(chunk)
                np_cause = str(chunk)
                for child in chunk.root.children:
                    if child.dep_ == "acl":
                        cause = " ".join([cause, str(child)])

                    if child.dep_ == "prep" and child in np_preps:
                        prep_cause = child

        # CAUSE PREPOSITION LINKING

        if prep_effect is not None or prep_cause is not None:

            cause, effect = self.prep_linking(
                prep_effect, prep_cause, doc, cause, effect
            )
        if self.np_link:
            if np_cause and np_cause == cause:
                cause = self.np_linker(cause, np_cause, np_list, doc)
            if np_effect and np_cause and np_effect == effect:
                effect = self.np_linker(effect, np_cause, np_list, doc, 1)

        if not effect and cause:
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"] and token.head.text == ROOT:
                    if "NOUN" in [
                        child.pos_ for child in token.children
                    ] and token.pos_ in ["ADV", "prep"]:
                        effect = " ".join(a.text for a in token.subtree)

        return cause, effect

    def primary_triplet_extraction(self, sentence, ROOT_KEYWORD, comp):

        doc = self.nlp(sentence)

        cause = ""
        effect = ""
        prep_cause = None
        prep_effect = None
        np_cause = ""
        np_effect = ""
        np_preps = []
        np_list = []
        ticker = 0
        for token in doc:

            if token.dep_ == "ROOT":
                if token.text.lower() != ROOT_KEYWORD:
                    for child in token.children:
                        if (
                            child.dep_ in ["acl", "advcl"]
                            and child.text == ROOT_KEYWORD
                        ):
                            ROOT = token.text
                            return self.secondary_triplet_extraction(
                                doc, ROOT_KEYWORD, comp, ROOT
                            )

            # EFFECT PREPOSITION

            if token.dep_ in ["nsubj", "nsubjpass"] and token.head.text == ROOT_KEYWORD:

                for child in token.children:
                    if child.dep_ == "prep":
                        prep_effect = child

        for chunk in doc.noun_chunks:
            np_list.append(chunk.text)
            if chunk.root.head.dep_ == "prep":
                np_preps.append(chunk.root.head)

        for chunk in doc.noun_chunks:

            if (
                chunk.root.dep_ in ["nsubj", "nsubjpass"]
                and chunk.root.head.text == ROOT_KEYWORD
            ):

                effect = str(chunk)
                np_effect = str(chunk)

            if chunk.root.head.text == comp:
                cause = str(chunk)
                np_cause = str(chunk)
                for child in chunk.root.children:
                    if child.dep_ == "acl":
                        cause = " ".join([cause, str(child)])

                    if child.dep_ == "prep" and child in np_preps:

                        prep_cause = child

        if prep_effect is not None or prep_cause is not None:

            cause, effect = self.prep_linking(
                prep_effect, prep_cause, doc, cause, effect
            )
        if self.np_link:
            if np_cause and np_cause == cause:
                cause = self.np_linker(cause, np_cause, np_list, doc)
            if np_effect and np_effect == effect:
                effect = self.np_linker(effect, np_cause, np_list, doc, 1)

        if not cause and not effect:
            return self.secondary_triplet_extraction(
                doc, ROOT_KEYWORD, comp, ROOT_KEYWORD
            )

        return cause, effect

    def np_linker(self, cause_effect, np_link, np_list, doc, type_=0):
        new_cause_effect = cause_effect
        if type_ == 0:
            if cause_effect in np_list and np_list.index(cause_effect) < len(np_list):
                for nc in doc.noun_chunks:
                    if np_list.index(nc.text) > np_list.index(cause_effect):
                        if nc.root.head.dep_ in ["prep", "relcl"]:
                            new_cause_effect = (
                                new_cause_effect + f" {nc.root.head.text} {nc.text}"
                            )
        elif type_ == 1:
            if cause_effect in np_list and np_list.index(cause_effect) < np_list.index(
                np_link
            ):
                for nc in doc.noun_chunks:
                    if np_list.index(nc.text) > np_list.index(
                        cause_effect
                    ) and np_list.index(nc.text) < np_list.index(np_link):
                        if nc.root.head.dep_ in ["prep", "relcl"]:
                            new_cause_effect = (
                                new_cause_effect + f" {nc.root.head.text} {nc.text}"
                            )
        return new_cause_effect
