from constant import invalid_relations_set
from statistics import mean
import spacy

def Map(head, relations, tail,nlp, top_first=True, best_scores=True):
    if head == None or tail == None or relations == None:
        return {}
    if len(head.split()) == 1:
        if nlp(head)[0].pos_ in ["PRON", "AUX"]:
            return {}
    if len(tail.split()) == 1:
        if nlp(tail)[0].pos_ in ["PRON", "AUX"]:
            return {}
    valid_relations = [
        r
        for r in relations
        if r not in invalid_relations_set and r.isalpha() and len(r) > 1
    ]
    if len(valid_relations) == 0:
        return {}

    return {"h": head, "t": tail, "r": "_".join(valid_relations)}


def deduplication(triplets):
    unique_pairs = []
    relations = []
    pair_confidence = []
    sentences = []
    for i, t in enumerate(triplets):
        key = "{}\t{}".format(t["h"], t["t"])
        relation = t["r"]
        confidence = t["c"]
        sent = t["sent"]
        if key in unique_pairs:
            idx = unique_pairs.index(key)
            rel = relations[idx]
            rel.append(relation)
            conf = pair_confidence[idx]
            conf.append(confidence)
        else:
            unique_pairs.append(key)
            rel = [relation]
            relations.append(rel)
            conf = [confidence]
            pair_confidence.append(conf)
            sentences.append(sent)
    unique_triplets = []
    for unique_pair, r, conf, sent in zip(
        unique_pairs, relations, pair_confidence, sentences
    ):
        if len(r) > 3:
            continue
        h, t = unique_pair.split("\t")
        unique_triplets.append(
            {"h": h, "r": [*{*r}], "t": t, "c": mean(conf), "sent": sent}
        )

    return unique_triplets


if __name__ == "__main__":
    pass
