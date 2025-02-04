# coding=utf-8

# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

"""Tokenization classes."""

from __future__ import absolute_import, division, print_function

import collections
import unicodedata

import six


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        ids.append(vocab[token])
    return ids


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, tokenize_method, do_lower_case=True, do_basic_tokenize=True):
        self.vocab = load_vocab(vocab_file)
        #FullTokenizer.add_tokens(['<number>', '<date>', '<time>', '<percent>'], self.vocab)

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.do_basic_tokenize = do_basic_tokenize
        self.tokenize_method = tokenize_method
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, tokenize_method=self.tokenize_method)

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            for sub_token in self.wordpiece_tokenizer.tokenize(text):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)

    @staticmethod
    def add_tokens(tokens,vocab):
        last_index = max(vocab.values()) +1
        for i,token in enumerate(tokens):
            new_index = last_index + i
            vocab[token] = new_index


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, tokenize_method, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.tokenize_method = tokenize_method


    def tokenize(self, text):
        text = convert_to_unicode(text)
        output_tokens = []
        # replace unknow word with the longest prefix match word in dictionary
        if self.tokenize_method == "prefix_match":
            for token in whitespace_tokenize(text):
                chars = list(token)
                if len(chars) > self.max_input_chars_per_word:
                    output_tokens.append(self.unk_token)
                    continue
                start = 0
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.append(cur_substr)
        # replace unknow word to [UNK]
        elif self.tokenize_method == "unk_replace":
            for token in whitespace_tokenize(text):
                chars = list(token)
                if len(chars) > self.max_input_chars_per_word:
                    output_tokens.append(self.unk_token)
                    continue
                temp_str = "".join(chars[0:])
                if temp_str in self.vocab:
                    output_tokens.append(temp_str)
                else:
                    output_tokens.append(self.unk_token)
        # split unknown word to several words
        elif self.tokenize_method == "word_split":
            for token in whitespace_tokenize(text):
                chars = list(token)
                if len(chars) > self.max_input_chars_per_word:
                    output_tokens.append(self.unk_token)
                    continue

                is_bad = False
                start = 0
                sub_tokens = []
                while start < len(chars):
                    end = len(chars)
                    cur_substr = None
                    while start < end:
                        substr = "".join(chars[start:end])
                        if start > 0:
                            substr = "##" + substr
                        if substr in self.vocab:
                            cur_substr = substr
                            break
                        end -= 1
                    if cur_substr is None:
                        is_bad = True
                        break
                    sub_tokens.append(cur_substr)
                    start = end

                if is_bad:
                    output_tokens.append(self.unk_token)
                else:
                    output_tokens.extend(sub_tokens)

        return output_tokens

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
