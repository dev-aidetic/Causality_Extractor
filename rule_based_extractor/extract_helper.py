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

    def passive_triplet_extraction(self, sent, ROOT_KEYWORD, comp="in"):
        doc = self.nlp(sent)
        cause = ""
        effect = ""
        prep_cause = None
        prep_effect = None
        np_effect = ""
        np_cause = ""
        np_preps = []
        np_list = []
        for token in doc:

            if token.dep_ == "ROOT":
                if token.text.lower() != ROOT_KEYWORD:
                    return cause, effect

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
                chunk.root.dep_ in ["nsubj", "nsubjpass"]
                and chunk.root.head.text == ROOT_KEYWORD
            ):

                cause = str(chunk)
                np_cause = str(chunk)

            if chunk.root.head.text in comp:
                effect = str(chunk)
                np_effect = str(chunk)
                for child in chunk.root.children:
                    if child.dep_ == "acl":
                        effect = " ".join([effect, str(child)])

                    if child.dep_ == "prep" and child in np_preps:
                        prep_effect = child

        # CAUSE PREPOSITION LINKING

        if prep_cause is not None or prep_effect is not None:

            effect, cause = self.prep_linking(
                prep_cause, prep_effect, doc, effect, cause
            )
        if self.np_link:
            if np_effect and np_effect == effect:
                effect = self.np_linker(effect, np_effect, np_list, doc)
            if np_cause and np_cause == cause:
                cause = self.np_linker(cause, np_effect, np_list, doc, 1)

        return cause, effect

    def active_triplet_extraction(self, sentence, ROOT_KEYWORD, comp="by"):

        doc = self.nlp(sentence)

        cause = ""
        effect = ""
        prep_cause = None
        prep_effect = None
        np_cause = ""
        np_effect = ""
        np_preps = []
        np_list = []

        for token in doc:

            if token.dep_ == "ROOT":
                if token.text.lower() != ROOT_KEYWORD:
                    return cause, effect

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

                    if (
                        child.dep_ == "prep" and child in np_preps
                    ):  # To handle multiple child preps to the root of the cause noun chunk

                        prep_cause = child

        # EFFECT PREPOSITION LINKING

        if prep_effect is not None or prep_cause is not None:

            cause, effect = self.prep_linking(
                prep_effect, prep_cause, doc, cause, effect
            )
        if self.np_link:
            if np_cause and np_cause == cause:
                cause = self.np_linker(cause, np_cause, np_list, doc)
            if np_effect and np_effect == effect:
                effect = self.np_linker(effect, np_cause, np_list, doc, 1)

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
