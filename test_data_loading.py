from processor import Semeval_Processor
import tokenization
from tqdm import tqdm, trange


def _truncate_seq_pair(tokens_a, tokens_b, ner_labels_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ner_labels_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ner_label_ids, ner_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ner_label_ids = ner_label_ids
        self.ner_mask = ner_mask

    def __str__(self):
        return f"input_ids: {self.input_ids}" \
               f"input_mask: {self.input_mask}" \
               f"segment_ids: {self.segment_ids}"


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, ner_label_list, tokenize_method):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # here start with zero this means that "[PAD]" is zero
    ner_label_map = {}
    for (i, label) in enumerate(ner_label_list):
        ner_label_map[label] = i

    features = []
    all_tokens = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if tokenize_method == "word_split":
            # word_split
            word_num = 0
            tokens_a = tokenizer.tokenize(example.text_a)
            ner_labels_org = example.ner_labels_a.strip().split()
            ner_labels_a = []
            token_bias_num = 0

            for (i, token) in enumerate(tokens_a):
                if token.startswith('##'):
                    if ner_labels_org[i - 1 - token_bias_num] in ['O', 'T', 'I']:
                        ner_labels_a.append(ner_labels_org[i - 1 - token_bias_num])
                    else:
                        ner_labels_a.append('I')
                    token_bias_num += 1
                else:
                    word_num += 1
                    ner_labels_a.append(ner_labels_org[i - token_bias_num])

            assert word_num == len(ner_labels_org)
            assert len(ner_labels_a) == len(tokens_a)

        else:
            # prefix_match or unk_replace
            tokens_a = tokenizer.tokenize(example.text_a)
            ner_labels_a = example.ner_labels_a.strip().split()

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, ner_labels_a, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
                ner_labels_a = ner_labels_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        ner_label_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        ner_label_ids.append(ner_label_map["[CLS]"])
        try:
            for (i, token) in enumerate(tokens_a):
                tokens.append(token)
                segment_ids.append(0)
                ner_label_ids.append(ner_label_map[ner_labels_a[i]])
        except:
            print(tokens_a)
            print(ner_labels_a)

        ner_mask = [1] * len(ner_label_ids)
        token_length = len(tokens)
        tokens.append("[SEP]")
        segment_ids.append(0)
        ner_label_ids.append(ner_label_map["[PAD]"])

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
                ner_label_ids.append(ner_label_map["[PAD]"])
            tokens.append("[SEP]")
            segment_ids.append(1)
            ner_label_ids.append(ner_label_map["[PAD]"])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            ner_label_ids.append(ner_label_map["[PAD]"])
        while len(ner_mask) < max_seq_length:
            ner_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(ner_mask) == max_seq_length
        assert len(ner_label_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                ner_label_ids=ner_label_ids,
                ner_mask=ner_mask))
        all_tokens.append(tokens[0:token_length])
    return features, all_tokens


tokenizer = tokenization.FullTokenizer(
    vocab_file="uncased_L-12_H-768_A-12/vocab.txt", tokenize_method="word_split", do_lower_case=True)

processor = Semeval_Processor()

train_examples = processor.get_train_examples("data/semeval2015/three_joint/BIO/")

label_list = processor.get_labels()
ner_label_list = processor.get_ner_labels("data/semeval2015/three_joint/BIO/")  # BIO or TO tags for ner entity

train_features, _ = convert_examples_to_features(
    train_examples, label_list, 128, tokenizer, ner_label_list, "word_split")

for train_feature in train_features:
    print(train_feature)