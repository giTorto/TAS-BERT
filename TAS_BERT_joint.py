# coding=utf-8

"""CUDA_VISIBLE_DEVICES=0 python TAS_BERT_joint.py"""
from __future__ import absolute_import, division, print_function

import copy

"""
three-joint detection for target & aspect & sentiment
"""

import argparse
import collections
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

import tokenization
from modeling import BertConfig, BertForTABSAJoint, BertForTABSAJoint_CRF
from optimization import BERTAdam

import datetime

from processor import Semeval_Processor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_arg_parse():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='data/semeval2015/three_joint/TO/',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default='results/semeval2015/three_joint/TO/my_result',
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--vocab_file",
                        default='uncased_L-12_H-768_A-12/vocab.txt',
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--bert_config_file",
                        default='uncased_L-12_H-768_A-12/bert_config.json',
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--init_checkpoint",
                        default='uncased_L-12_H-768_A-12/pytorch_model.bin',
                        type=str,
                        required=True,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--tokenize_method",
                        default='word_split',
                        type=str,
                        required=True,
                        choices=["prefix_match", "unk_replace", "word_split"],
                        help="how to solve the unknow words, max prefix match or replace with [UNK] or split to some words")
    parser.add_argument("--use_crf",
                        default=True,
                        required=True,
                        action='store_true',
                        help="Whether to use CRF after Bert sequence_output")

    ## Other parameters
    parser.add_argument("--eval_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--alberto",
                        default=False,
                        action='store_true',
                        help="Whether the model uses alberto")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--patience",
                        type=int,
                        default=5,
                        help="The patience parameter for early stopping")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    return parser


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ner_label_ids, ner_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ner_label_ids = ner_label_ids
        self.ner_mask = ner_mask


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
    all_ner_masks = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if tokenize_method == "word_split":
            mask_org = example.ner_mask
            # word_split
            word_num = 0
            tokens_a = tokenizer.tokenize(example.text_a)
            ner_labels_org = example.ner_labels_a.strip().split()
            ner_labels_a = []
            ner_mask_a = []
            token_bias_num = 0

            for (i, token) in enumerate(tokens_a):
                if token.startswith('##'):
                    if ner_labels_org[i - 1 - token_bias_num] in ['O', 'T', 'I']:
                        ner_labels_a.append(ner_labels_org[i - 1 - token_bias_num])
                        ner_mask_a.append(mask_org[i - 1 - token_bias_num])
                    else:
                        ner_labels_a.append('I')
                        ner_mask_a.append(1)
                    token_bias_num += 1
                else:
                    word_num += 1
                    ner_labels_a.append(ner_labels_org[i - token_bias_num])
                    ner_mask_a.append(mask_org[i - token_bias_num])
            assert word_num == len(ner_labels_org)
            assert len(ner_labels_a) == len(tokens_a)

        else:
            # prefix_match or unk_replace
            tokens_a = tokenizer.tokenize(example.text_a)
            ner_labels_a = example.ner_labels_a.strip().split()
            ner_mask_a = example.ner_mask

        # print(f"Len of tokens_a == ner_label_a - {len(tokens_a)} vs {len(ner_labels_a)}")
        # print(f"Len of tokens_a == ner_label_a - {tokens_a} vs {ner_labels_a}")

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, ner_labels_a, max_seq_length - 3, ner_mask=ner_mask_a)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
                ner_labels_a = ner_labels_a[0:(max_seq_length - 2)]
                ner_mask_a = ner_mask_a[0:(max_seq_length - 2)]

        # print(f"Len of tokens_a == ner_label_a - {len(tokens_a)} vs {len(ner_labels_a)}")
        # print(f"Len of tokens_a == ner_label_a - {tokens_a} vs {ner_labels_a}")

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
        ner_mask_a.insert(0, 1)
        try:
            for (i, token) in enumerate(tokens_a):
                tokens.append(token)
                segment_ids.append(0)
                ner_label_ids.append(ner_label_map[ner_labels_a[i]])
        except Exception as e:
            print(tokens_a)
            print(ner_labels_a)
            print(ner_label_map)
            raise e

        # print(f"Len of tokens == ner_label_ids - {len(tokens)} vs {len(ner_label_ids)}")
        # print(f"Len of tokens == ner_label_ids - {tokens} vs {ner_label_ids}")

        ner_mask = ner_mask_a
        token_length = len(tokens)
        tokens.append("[SEP]")
        segment_ids.append(0)
        ner_label_ids.append(ner_label_map["[PAD]"])

        # print(f"Len of tokens == ner_label_ids - {len(tokens)} vs {len(ner_label_ids)}")
        # print(f"Len of tokens == ner_label_ids - {tokens} vs {ner_label_ids}")

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
                ner_label_ids.append(ner_label_map["[PAD]"])
            tokens.append("[SEP]")
            segment_ids.append(1)
            ner_label_ids.append(ner_label_map["[PAD]"])

        # print(f"Len of tokens == ner_label_ids - {len(tokens)} vs {len(ner_label_ids)}")
        # print(f"Len of tokens == ner_label_ids - {tokens} vs {ner_label_ids}")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(f"Len of input ids == ner_label_ids - {len(input_ids)} vs {len(ner_label_ids)}")

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

        # print(f"Len of input ids == ner_label_ids - {len(input_ids)} vs {len(ner_label_ids)}")
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


def _truncate_seq_pair(tokens_a, tokens_b, ner_labels_a, max_length, ner_mask):
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
            ner_mask.pop()
        else:
            tokens_b.pop()


def convert_alberto_state_dict_to_bert(alberto_state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in alberto_state_dict.items():
        new_key = key
        if key.startswith("bert."):
            new_key = key.replace("bert.", "", 1)

        new_key = new_key.replace("LayerNorm.bias", "LayerNorm.beta")
        new_key = new_key.replace("LayerNorm.weight", "LayerNorm.gamma")

        if new_key.startswith("cls."):
            continue
        new_state_dict[new_key] = value
    return new_state_dict


def create_data_loader(data_dir, tokenize_method, max_seq_length, tokenizer, ner_label_list, label_list, processor,
                       batch_size, local_rank=None, data_type="train"):
    if data_type == "train":
        data_examples = processor.get_train_examples(data_dir)
    elif data_type == "dev":
        data_examples = processor.get_dev_examples(data_dir)
    else:
        data_examples = processor.get_test_examples(data_dir)

    data_features, data_tokens = convert_examples_to_features(
        data_examples, label_list, max_seq_length, tokenizer, ner_label_list, tokenize_method)

    all_input_ids = torch.tensor([f.input_ids for f in data_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in data_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in data_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in data_features], dtype=torch.long)
    all_ner_label_ids = torch.tensor([f.ner_label_ids for f in data_features], dtype=torch.long)
    all_ner_mask = torch.tensor([f.ner_mask for f in data_features], dtype=torch.long)

    input_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ner_label_ids,
                               all_ner_mask)
    sampler = None
    if data_type == "train":
        if local_rank == -1:
            sampler = RandomSampler(input_data)
        else:
            sampler = DistributedSampler(input_data)
    input_dataloader = DataLoader(input_data, sampler=sampler, batch_size=batch_size)
    return input_dataloader, data_tokens


def persist_prediction(output_dir, epoch, outputs, label_ids, ner_test_tokens, ner_label_list, ner_label_ids,
                       ner_logits):
    with open(os.path.join(output_dir, "test_ep_" + str(epoch) + ".txt"), "w") as f_test:
        f_test.write('yes_not\tyes_not_pre\tsentence\ttrue_ner\tpredict_ner\n')
        for output_i in range(len(outputs)):
            # category & polarity
            f_test.write(str(label_ids[output_i]))
            f_test.write('\t')
            f_test.write(str(outputs[output_i]))
            f_test.write('\t')

            # sentence & ner labels
            sentence_clean = []
            label_true = []
            label_pre = []
            sentence_len = len(ner_test_tokens[output_i])

            for i in range(sentence_len):
                if not ner_test_tokens[output_i][i].startswith('##'):
                    """
                    print("OUTPUT_I", output_i, "I", i, "\n",
                          "ner_test_tokens[output_i]", ner_test_tokens[output_i], len(ner_test_tokens[output_i]), "\n",
                          "ner_label_ids[output_i]", ner_label_ids[output_i],len(ner_label_ids[output_i]), "\n",
                          "ner_logits[output_i]", ner_logits[output_i], len(ner_logits[output_i]), "\n",
                    )"""
                    sentence_clean.append(ner_test_tokens[output_i][i])
                    label_true.append(ner_label_list[ner_label_ids[output_i][i]])
                    label_pre.append(ner_label_list[ner_logits[output_i][i]])

            f_test.write(' '.join(sentence_clean))
            f_test.write('\t')
            f_test.write(' '.join(label_true))
            f_test.write("\t")
            f_test.write(' '.join(label_pre))
            f_test.write("\n")


def predict(test_dataloader, test_tokens, ner_label_list, model, epoch, device, output_dir, use_crf, eval_batch_size,
            do_persist=False):
    model.eval()
    test_loss, test_accuracy = 0, 0
    ner_test_loss = 0
    nb_test_steps, nb_test_examples = 0, 0
    batch_index = 0
    for input_ids, input_mask, segment_ids, label_ids, ner_label_ids, ner_mask in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        ner_label_ids = ner_label_ids.to(device)
        ner_mask = ner_mask.to(device)
        # test_tokens is the origin word in sentences [batch_size, sequence_length]
        ner_test_tokens = test_tokens[batch_index * eval_batch_size:(batch_index + 1) * eval_batch_size]
        batch_index += 1

        with torch.no_grad():
            tmp_test_loss, tmp_ner_test_loss, logits, ner_predict = model(input_ids, segment_ids, input_mask,
                                                                          label_ids, ner_label_ids, ner_mask)

        # category & polarity
        logits = F.softmax(logits, dim=-1)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        outputs = np.argmax(logits, axis=1)

        if use_crf:
            # CRF
            ner_logits = ner_predict
        else:
            # softmax
            ner_logits = torch.argmax(F.log_softmax(ner_predict, dim=2), dim=2)
            ner_logits = ner_logits.detach().cpu().numpy()

        ner_label_ids = ner_label_ids.to('cpu').numpy()
        ner_mask = ner_mask.to('cpu').numpy()

        if do_persist:
            persist_prediction(output_dir, epoch, outputs, label_ids, ner_test_tokens, ner_label_list, ner_label_ids,
                               ner_logits)
        tmp_test_accuracy = np.sum(outputs == label_ids)
        test_loss += tmp_test_loss.mean().item()
        ner_test_loss += tmp_ner_test_loss.mean().item()
        test_accuracy += tmp_test_accuracy

        nb_test_examples += input_ids.size(0)
        nb_test_steps += 1

    test_loss = test_loss / nb_test_steps
    ner_test_loss = ner_test_loss / nb_test_steps
    test_accuracy = test_accuracy / nb_test_examples

    return test_loss, ner_test_loss, test_accuracy


def train_step(train_dataloader, device, model, n_gpu, gradient_accumulation_steps, optimizer, global_step):
    model.train()

    tr_loss = 0
    tr_ner_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, ner_label_ids, ner_mask = batch
        loss, ner_loss, _, _ = model(input_ids, segment_ids, input_mask, label_ids, ner_label_ids, ner_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
            ner_loss = ner_loss.mean()
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            ner_loss = ner_loss / gradient_accumulation_steps
        loss.backward(retain_graph=True)
        ner_loss.backward()

        tr_loss += loss.item()
        tr_ner_loss += ner_loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()  # We have accumulated enought gradients
            model.zero_grad()
            global_step += 1

    result = {
     'global_step': global_step,
     'loss': tr_loss / nb_tr_steps,
     'ner_loss': tr_ner_loss / nb_tr_steps}

    return result


def init_optimizer(model, num_train_steps, learning_rate, warmup_proportion):
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)

    return optimizer


def init_components(bert_config_file, max_seq_length, vocab_file, data_dir, tokenize_method, do_lower_case, use_crf,
                    init_checkpoint, device, alberto, local_rank, n_gpu):
    bert_config = BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                max_seq_length, bert_config.max_position_embeddings))

    processor = Semeval_Processor()
    label_list = processor.get_labels()
    ner_label_list = processor.get_ner_labels(data_dir)  # BIO or TO tags for ner entity

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, tokenize_method=tokenize_method, do_lower_case=do_lower_case,
        do_basic_tokenize=False)

    if use_crf:
        model = BertForTABSAJoint_CRF(bert_config, len(label_list), len(ner_label_list))
    else:
        model = BertForTABSAJoint(bert_config, len(label_list), len(ner_label_list), max_seq_length)

    if init_checkpoint is not None:
        state_dict = torch.load(init_checkpoint, map_location='cpu')
        if alberto:
            state_dict = convert_alberto_state_dict_to_bert(state_dict)
        model.bert.load_state_dict(state_dict)
    model.to(device)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, processor, tokenizer, label_list, ner_label_list


def compute_best_epoch(last_best, current_epoch, metric="loss", target="min"):
    """

    :param last_best: The last best epoch metrics
    :param current_epoch: the current epoch metrics
    :param metric: the type of metric to use
    :param target: the target whether to "maximize"= "max" or "minimize" = "min"
    :return:
        the best epoch, changed - the new dict and whether the
    """

    last_best_value = last_best.get(metric, 0.0 if target == "max" else 9999999.9)
    current_epoch_value = current_epoch.get(metric)

    if target == "max":
        return (last_best, False) if last_best_value > current_epoch_value else (current_epoch, True)
    else:
        return (last_best, False) if last_best_value < current_epoch_value else (current_epoch, True)


def train(model, train_dataloader, optimizer, patience, num_train_epochs, output_dir, eval_test, n_gpu, device,
          gradient_accumulation_steps, dev_dataloader, dev_tokens, ner_label_list, eval_batch_size, use_crf):
    output_log_file = os.path.join(output_dir, "log.txt")
    print("output_log_file=", output_log_file)
    with open(output_log_file, "w") as writer:
        if eval_test:
            writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
        else:
            writer.write("epoch\tglobal_step\tloss\n")

    checkpoint_folder = "./checkpoints"

    global_step = 0
    epoch = 0
    last_best = {}
    patience = patience
    changes_counter = 0
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        epoch += 1

        result = \
            train_step(train_dataloader, device, model, n_gpu, gradient_accumulation_steps, optimizer, global_step)

        result['epoch'] = epoch

        # eval_test
        if eval_test:
            dev_loss, ner_dev_loss, dev_accuracy = \
                predict(dev_dataloader, dev_tokens, ner_label_list, model, epoch, device, output_dir,
                        use_crf, eval_batch_size, do_persist=False)

            result.update({
                'dev_loss': dev_loss,
                'ner_dev_loss': ner_dev_loss,
                'dev_accuracy': dev_accuracy
            })

        prev_best = copy.copy(last_best)
        last_best, changed = compute_best_epoch(last_best, result)

        if not changed:
            changes_counter += 1
        else:
            out_path = os.path.join(checkpoint_folder, "epoch_" + str(epoch))
            changes_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': result.get("loss"),
                'path': out_path
            }, out_path)
            if prev_best.get("path") is not None:
                os.remove(prev_best.get("path"))

        logger.info(f"Last best: {last_best} and changes_counter:{changes_counter}")
        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")

        if changes_counter >= patience:
            break

    return last_best


def main():
    parser = create_arg_parse()
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, processor, tokenizer, label_list, ner_label_list = \
        init_components(args.bert_config_file, args.max_seq_length, args.vocab_file, args.data_dir,
                        args.tokenize_method,
                        args.do_lower_case, args.use_crf,
                        args.init_checkpoint, device, args.alberto, args.local_rank, n_gpu)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, tokenize_method=args.tokenize_method, do_lower_case=args.do_lower_case,
        do_basic_tokenize=False)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # training set
    train_dataloader, _ = create_data_loader(
        args.data_dir, args.tokenize_method, args.max_seq_length, tokenizer, ner_label_list, label_list,
        processor, args.eval_batch_size, data_type="train", local_rank=args.local_rank)
    dev_dataloader, dev_tokens = create_data_loader(
        args.data_dir, args.tokenize_method, args.max_seq_length, tokenizer, ner_label_list, label_list,
        processor, args.eval_batch_size, data_type="dev")
    test_dataloader, test_tokens = create_data_loader(
        args.data_dir, args.tokenize_method, args.max_seq_length, tokenizer, ner_label_list, label_list,
        processor, args.eval_batch_size, data_type="test")

    num_train_steps = int(
        len(train_dataloader) / args.train_batch_size * args.num_train_epochs)

    optimizer = init_optimizer(model, num_train_steps, args.learning_rate, args.warmup_proportion)

    # train
    last_best = train(model, train_dataloader, optimizer, args.patience, args.num_train_epochs, args.output_dir, args.eval_test, n_gpu, device,
          args.gradient_accumulation_steps, dev_dataloader, dev_tokens, ner_label_list, args.eval_batch_size, args.use_crf)

    final_model = last_best.get("path")
    with open(final_model, 'r') as in_file:
        checkpoint = torch.load(final_model)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    test_loss, ner_test_loss, test_accuracy = \
        predict(test_dataloader, test_tokens, ner_label_list, model, epoch, device, args.output_dir,
                args.use_crf, args.eval_batch_size, do_persist=True)


if __name__ == "__main__":
    main()
