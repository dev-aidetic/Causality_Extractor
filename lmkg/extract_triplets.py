from process import parse_sentence
from mapper import Map, deduplication
from transformers import AutoTokenizer, BertModel, GPT2Model
import os


def ExtractTriplets(sentences, tokenizer, encoder, nlp, use_cuda, threshold):
    if not use_cuda:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()

    output = []
    if len(sentences):
        for idx, sent in enumerate(sentences):
            # Match
            valid_triplets = []
            for triplets in parse_sentence(
                sent, tokenizer, encoder, nlp, use_cuda=use_cuda
            ):
                triplets["l"] = idx
                valid_triplets.append(triplets)
            if len(valid_triplets) > 0:
                # Map
                mapped_triplets = []
                for triplet in valid_triplets:
                    head = triplet["h"]
                    tail = triplet["t"]
                    relations = triplet["r"]
                    conf = triplet["c"]
                    line = triplet["l"]
                    if conf < threshold:
                        continue
                    mapped_triplet = Map(head, relations, tail,nlp)
                    if "h" in mapped_triplet:
                        mapped_triplet["c"] = conf
                        mapped_triplet["sent"] = sentences[line]
                        mapped_triplets.append(mapped_triplet)
                output.extend(deduplication(mapped_triplets))
    print(output)
    return output


if __name__ == "__main__":
    pass
