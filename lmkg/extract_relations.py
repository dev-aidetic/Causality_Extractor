from extract_triplets import ExtractTriplets
from has_cause_mapper import MapHasCauseRelations
from complete_head_tails import GetCompleteHeadTails

def ExtractRelations(
    sentences,
    tokenizer,
    encoder,
    nlp,
    use_cuda=False,
    threshold=0.003,
):
    lmkg_triplets = ExtractTriplets(
        sentences, tokenizer, encoder, nlp, use_cuda, threshold
    )

    wiki_data_mapped_data = MapHasCauseRelations(lmkg_triplets,nlp)

    mapped_only_data = []
    for triplet in wiki_data_mapped_data:
        if "mapped_relations" in triplet:
            if len(triplet["mapped_relations"]["wiki_data_relation"]) > 0:
                mapped_only_data.append(triplet)

    final_mapping_data = GetCompleteHeadTails(mapped_only_data, nlp)

    return final_mapping_data


if __name__ == "__main__":
    pass