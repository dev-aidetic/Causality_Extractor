import contractions
import re

from numpy.core.numeric import NaN


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


mapper = {
    "caused": "has cause",
    "driven": "has cause",
    "impacted": "has cause",
    "informed": "has cause",
    "amplified": "has cause",
    "affected": "has cause",
    "offset": "has effect",
    "resulting": "has effect",
    "results": "has effect",
    "result": "has effect",
    "resulted": "has effect",
    "causing": "has effect",
    "led": "has effect",
    "leads": "has effect",
    "leading": "has effect",
}

"""
def rearrange_result(sentences, lmkg_result, rule_based_result):

    final_result = []

    lmkg_result = {result["sentence"]: result for result in lmkg_result}
    rule_based_result = {result["sentence"]: result for result in rule_based_result}
    added_list = []
    for sentence in sentences:

        sentence = clean_text(sentence)
        if sentence in lmkg_result:
            added_list.append(sentence)
            final_result.append(lmkg_result[sentence])
        if sentence in rule_based_result:
            added_list.append(sentence)
            final_result.append(rule_based_result[sentence])

    added_list = list(set(added_list))

    for sentence in lmkg_result:
        if sentence not in added_list:
            final_result.append(lmkg_result[sentence])

    for sentence in rule_based_result:
        if sentence not in added_list:
            final_result.append(rule_based_result[sentence])

    return final_result
"""


def rearrange_result(lmkg_result, rule_based_result):
    result_lmkg = []
    result_rule_based = []
    if lmkg_result:
        result_lmkg = [
            {
                "sentence": rel["sent"],
                "head": rel["h"],
                "raw_relation": rel["r"],
                "tail": rel["t"],
                "relation": rel["wikidata_relation"],
                "relation_score": rel["wikidata_relation_score"],
            }
            for rel in lmkg_result["relations"]
            if "relations" in lmkg_result
        ]
    if rule_based_result:

        for rel in rule_based_result:
            rel_dict = {}
            rel_dict["sentence"] = rel["sentence"]

            if mapper[rel["relation"]] == "has effect":
                rel_dict["head"] = rel["cause"]
                rel_dict["raw_relation"] = [rel["relation"]]
                rel_dict["tail"] = rel["effect"]
            else:
                rel_dict["head"] = rel["effect"]
                rel_dict["raw_relation"] = [rel["relation"]]
                rel_dict["tail"] = rel["cause"]
            rel_dict["relation"] = [mapper[rel["relation"]]]
            rel_dict["relation_score"] = None
            result_rule_based.append(rel_dict)

    return result_lmkg, result_rule_based
