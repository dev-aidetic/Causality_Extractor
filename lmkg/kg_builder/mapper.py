from kg_builder.constant import invalid_relations_set


def Map(head, relations, tail, top_first=True, best_scores=True):
    if head == None or head == "" or tail == None or tail == "" or relations == None:
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
    pair_confidence = []
    duplicates = {}
    for t in triplets:
        key = "{}\t{}\t{}".format(t["h"], t["r"], t["t"])
        h_t_pair = "{}\t{}".format(t["h"], t["t"])
        conf = t["c"]
        if h_t_pair in duplicates:
            duplicates[h_t_pair].append(key)
        else:
            duplicates[h_t_pair] = [key]
        if key not in unique_pairs:
            unique_pairs.append(key)
            pair_confidence.append(conf)

    unique_triplets = []

    for key, values in duplicates.items():
        relaions = []
        conf = []
        for value in values:
            h, r, t = value.split("\t")
            relaions.append(r)
            conf.append(pair_confidence[unique_pairs.index(value)])
        if len(set(relaions)) < 4:
            unique_triplets.append(
                {
                    "h": h,
                    "r": list(set(relaions)),
                    "t": t,
                    "c": round(sum(conf) / len(conf), 2),
                }
            )

    return unique_triplets
