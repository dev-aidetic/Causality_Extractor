import contractions
import re
import json

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


def GetCompleteHeadTails(mapped_data, nlp):
    sent_heads_dict = dict()
    sent_tails_dict = dict()
    sent_mapped_rel_dict = dict()
    c_score_dict = {}
    sim_score_dict = {}
    for data_dict in mapped_data:
        clean_sent = clean_text(data_dict["sent"])
        clean_head = clean_text(data_dict["h"])
        clean_tail = clean_text(data_dict["t"])
        if clean_sent in sent_heads_dict.keys():
            sent_heads_dict[clean_sent].append(clean_head)
            sent_tails_dict[clean_sent].append(clean_tail)
            c_score_dict[clean_sent]['sum'] += data_dict["c"]
            c_score_dict[clean_sent]['count'] += 1
            continue
        sim_score_dict[clean_sent] = data_dict["mapped_relations"]["similarity_score"]
        c_score_dict[clean_sent] = {"sum":data_dict["c"],"count":1}
        sent_heads_dict[clean_sent] = [clean_head]
        sent_tails_dict[clean_sent] = [clean_tail]
        sent_mapped_rel_dict[clean_sent] = data_dict["mapped_relations"][
            "wiki_data_relation"
        ][0]
    for clean_sent in c_score_dict:
        c_score_dict[clean_sent] = (c_score_dict[clean_sent]["sum"]/c_score_dict[clean_sent]["count"])
    final_data = []
    for sent in sent_heads_dict.keys():
        heads = [*{*sent_heads_dict[sent]}]
        tails = [*{*sent_tails_dict[sent]}]
        relation = sent_mapped_rel_dict[sent]
        doc = nlp(sent)
        cause = ""
        effect = ""
        if relation == "has cause":
            for chunk in doc.noun_chunks:
                head = ""
                token = chunk.root
                flag = 0
                if (
                    chunk.text in heads
                    and chunk.root.head.dep_ == "prep"
                    and chunk.root.head.head.dep_ in ["nsubj", "dobj"]
                ) or (
                    chunk.text in heads
                    and chunk.root.head.dep_ == "prep"
                    and chunk.root.head.head.pos_ == "NUM"
                ):
                    if (
                        chunk.root.head.head.text
                        in " ".join([h for h in heads]).split()
                    ):
                        continue
                    if token.is_ancestor:
                        for child in token.children:
                            if child.dep_ == "prep":
                                for ch in child.children:
                                    if ch.dep_ == "pobj":
                                        head += ch.text
                                        head = (
                                            " ".join(
                                                [
                                                    c.text
                                                    for c in ch.children
                                                    if c.dep_
                                                    in ["amod", "det", "compound"]
                                                ]
                                            )
                                            + " "
                                            + head
                                        )
                                head = child.text + " " + head
                    head = token.text + " " + head
                    head = (
                        " ".join(
                            [
                                c.text
                                for c in token.children
                                if c.dep_ in ["amod", "det", "compound"]
                            ]
                        )
                        + " "
                        + head
                    )
                    head = token.head.text + " " + head
                    flag = 1
                    token = token.head.head
                if chunk.text in heads and chunk.root.head.pos_ == "VERB":
                    if token.is_ancestor:
                        for child in token.children:
                            if child.dep_ == "prep":
                                for ch in child.children:
                                    if ch.dep_ == "pobj":
                                        head += ch.text
                                        head = (
                                            " ".join(
                                                [
                                                    c.text
                                                    for c in ch.children
                                                    if c.dep_
                                                    in ["amod", "det", "compound"]
                                                ]
                                            )
                                            + " "
                                            + head
                                        )
                                head = child.text + " " + head
                    flag = 1
                if flag == 1:
                    head = token.text + " " + head
                    head = (
                        " ".join(
                            [
                                c.text
                                for c in token.children
                                if c.dep_ in ["amod", "det", "compound"]
                            ]
                        )
                        + " "
                        + head
                    )
                    if not token.pos_ == "NUM":
                        head = token.head.text + " " + head
                        head = (
                            " ".join(
                                [
                                    c.text
                                    for c in token.head.children
                                    if c.dep_ in ["aux", "auxpass", "neg"]
                                ]
                            )
                            + " "
                            + head
                        )
                        if token.head.dep_ in ["ccomp", "xcomp"] and any(
                            [
                                True
                                for c in token.head.head.children
                                if c.dep_ in ["neg"]
                            ]
                        ):
                            head = token.head.head.text + " " + head
                            head = (
                                " ".join(
                                    [
                                        c.text
                                        for c in token.head.head.children
                                        if c.dep_ in ["neg"]
                                    ]
                                )
                                + " "
                                + head
                            )
                    if head not in effect:
                        effect = re.sub(r"\s+", " ", head) + ", " + effect
            for chunk in doc.noun_chunks:
                tail = ""
                token = chunk.root
                flag = 0
                if chunk.text in tails:
                    if token.is_ancestor:
                        for child in token.children:
                            if child.dep_ == "prep":
                                for ch in child.children:
                                    if ch.dep_ == "pobj":
                                        tail += ch.text
                                        tail = (
                                            " ".join(
                                                [
                                                    c.text
                                                    for c in ch.children
                                                    if c.dep_
                                                    in ["amod", "det", "compound"]
                                                ]
                                            )
                                            + " "
                                            + tail
                                        )
                                tail = child.text + " " + tail
                    tail = token.text + " " + tail
                    tail = (
                        " ".join(
                            [
                                c.text
                                for c in token.children
                                if c.dep_ in ["amod", "det", "compound"]
                            ]
                        )
                        + " "
                        + tail
                    )
                    if token.head.dep_ == "prep" and token.head.head.dep_ in [
                        "dobj",
                        "nsubj",
                        "pobj",
                    ]:
                        continue
                    if token.head.pos_ == "VERB" and any(
                        [
                            True
                            for c in token.head.children
                            if c.dep_ == "nsubj" and c.text != token.text
                        ]
                    ):
                        tail = (
                            " ".join(
                                [
                                    c.text
                                    for c in token.head.children
                                    if c.dep_ in ["advmod", "neg"]
                                ]
                            )
                            + " "
                            + token.head.text
                            + " "
                            + tail
                        )
                    if token.head.pos_ == "VERB" and not any(
                        [
                            True
                            for c in token.head.children
                            if c.dep_ == "nsubj" and c.text != token.text
                        ]
                    ):
                        tail += (
                            " "
                            + " ".join(
                                [
                                    c.text
                                    for c in token.head.children
                                    if c.dep_ in ["advmod", "neg"]
                                ]
                            )
                            + " "
                            + token.head.text
                        )
                    if tail not in cause:
                        cause = re.sub(r"\s+", " ", tail) + ", " + cause
        if relation == "has effect":
            for chunk in doc.noun_chunks:
                head = ""
                token = chunk.root
                flag = 0
                if chunk.text in heads:
                    if token.is_ancestor:
                        for child in token.children:
                            if child.dep_ == "prep":
                                for ch in child.children:
                                    if ch.dep_ == "pobj":
                                        head += ch.text
                                        head = (
                                            " ".join(
                                                [
                                                    c.text
                                                    for c in ch.children
                                                    if c.dep_
                                                    in ["amod", "det", "compound"]
                                                ]
                                            )
                                            + " "
                                            + head
                                        )
                                head = child.text + " " + head
                    head = token.text + " " + head
                    head = (
                        " ".join(
                            [
                                c.text
                                for c in token.children
                                if c.dep_ in ["amod", "det", "compound"]
                            ]
                        )
                        + " "
                        + head
                    )
                    if token.head.dep_ == "prep" and token.head.head.dep_ in [
                        "dobj",
                        "nsubj",
                        "pobj",
                    ]:
                        continue
                    if head not in cause:
                        cause = re.sub(r"\s+", " ", head) + ", " + cause
            for chunk in doc.noun_chunks:
                tail = ""
                token = chunk.root
                flag = 0
                if chunk.text in tails and chunk.root.head.head.pos_ == "VERB":
                    if token.is_ancestor:
                        for child in token.children:
                            if child.dep_ == "prep":
                                for ch in child.children:
                                    if ch.dep_ == "pobj":
                                        tail += ch.text
                                        tail = (
                                            " ".join(
                                                [
                                                    c.text
                                                    for c in ch.children
                                                    if c.dep_
                                                    in ["amod", "det", "compound"]
                                                ]
                                            )
                                            + " "
                                            + tail
                                        )
                                tail = child.text + " " + tail
                    tail = token.text + " " + tail
                    tail = (
                        " ".join(
                            [
                                c.text
                                for c in token.children
                                if c.dep_ in ["amod", "det", "compound"]
                            ]
                        )
                        + " "
                        + tail
                    )
                    if tail not in effect:
                        effect = re.sub(r"\s+", " ", tail) + ", " + effect
        new_dict = {
            "sentence": sent,
            "causes": cause,
            "effects": effect,
            "confidence":c_score_dict[sent],
            "similarity_score":sim_score_dict[sent]
        }
        final_data.append(new_dict)
    with open('final_data.json','w') as fp:
        json.dump(final_data,fp,indent=2)
    return final_data


if __name__ == "__main__":
    pass
