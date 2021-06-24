import contractions
import re

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


def rearrange_result(sentences,lmkg_result,rule_based_result):

    final_result = []

    lmkg_result = { result["sentence"]:result for result in lmkg_result}
    rule_based_result = { result["sentence"]:result for result in rule_based_result}

    {
          "sentence": "Our expanded product vision was driven by our customers' desire to leverage MongoDB more broadly across their organization.",
          "cause": "our customers' desire leverage",
          "effect": "Our expanded product vision",
          "relation": "driven"
        }
    {
          "sentence": "Our available capital increased to a new record high of 39.2 billion driven by additional commitments from our flagship fund families including corporate private equity special opportunities and alternative credit strategies.",
          "causes": "a record high of 39.2 billion, available capital , ",
          "effects": "alternative credit strategies , additional commitments including corporate private special from fund familiesopportunities, ",
          "confidence": 0.05161989899352193,
          "similarity_score": 1
        }
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



