# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import json

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, segment_ids, start_labels, end_labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.segment_ids = segment_ids
        self.start_labels = start_labels
        self.end_labels = end_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, start_label_ids, end_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_label_ids = start_label_ids
        self.end_label_ids = end_label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line=='\n' or line=='':
                continue
            line_json = json.loads(line)
            words = line_json['tokens']
            if mode=='test': 
                start_labels=['O']*len(words)
                end_labels = ['O']*len(words)
            else: 
                start_labels = line_json['start_labels']
                end_labels = line_json['end_labels']
            if len(words)!= len(start_labels) :
                print(words, start_labels," length misMatch")
                continue
            if len(words)!= len(end_labels) :
                print(words, end_labels," length misMatch")
                continue
            segment_ids= line_json["segment_ids"]

            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words,\
                 start_labels=start_labels, end_labels = end_labels, segment_ids= segment_ids))
            guid_index += 1
                
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    trigger_token_segment_id =1,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['O'] = -1
    # print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        # print(example.words, example.labels)
        # print(len(example.words), len(example.labels))
        tokens = []
        start_label_ids = []
        end_label_ids = []
        segment_ids = []
        for word, start_label, end_label, segment_id in zip(example.words, example.start_labels, example.end_labels, example.segment_ids):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            if len(word_tokens)==1:
                tokens.extend(word_tokens)
            if len(word_tokens)>1: 
                print(word,">1") 
                tokens.extend(word_tokens[:1])
                pass
            if len(word_tokens)<1: 
                # print(word,"<1") 基本都是空格
                tokens.extend(["[unused1]"])
                # continue
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            cur_start_labels = start_label.split()
            cur_start_label_ids = []
            for cur_start_label in cur_start_labels:
                cur_start_label_ids.append(label_map[cur_start_label])
            start_label_ids.append(cur_start_label_ids)

            cur_end_labels = end_label.split()
            cur_end_label_ids = []
            for cur_end_label in cur_end_labels:
                cur_end_label_ids.append(label_map[cur_end_label])
            end_label_ids.append(cur_end_label_ids)

            segment_ids.extend( [sequence_a_segment_id if not segment_id else trigger_token_segment_id] * 1)

            # if len(tokens)!= len(label_ids):
            #     print(word, word_tokens, tokens, label_ids)
        # print(len(tokens),len(label_ids))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_label_ids = start_label_ids[: (max_seq_length - special_tokens_count)]
            end_label_ids = end_label_ids[: (max_seq_length - special_tokens_count)]
            segment_ids = segment_ids[: (max_seq_length - special_tokens_count)]


        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        start_label_ids += [[pad_token_label_id]]
        end_label_ids += [[pad_token_label_id]]
        segment_ids += [sequence_a_segment_id]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            start_label_ids += [[pad_token_label_id]]
            end_label_ids += [[pad_token_label_id]]
            segment_ids += [sequence_a_segment_id]
        # segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            start_label_ids += [[pad_token_label_id]]
            end_label_ids += [[pad_token_label_id]]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_label_ids = [[pad_token_label_id]] + start_label_ids
            end_label_ids = [[pad_token_label_id]] + end_label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(len(tokens), len(input_ids), len(label_ids))

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_label_ids = ([[pad_token_label_id]] * padding_length) + start_label_ids
            end_label_ids = ([[pad_token_label_id]] * padding_length) + end_label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_label_ids += [[pad_token_label_id]] * padding_length
            end_label_ids += [[pad_token_label_id]] * padding_length
        
        # print(len(label_ids), max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_label_ids) == max_seq_length
        assert len(end_label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("start_label_ids: %s", " ".join([str(x) for x in start_label_ids]))
            logger.info("end_label_ids: %s", " ".join([str(x) for x in end_label_ids]))
        
        if sum(segment_ids)==0:
            print(ex_index, "segment_id == None")
            continue
        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, \
                start_label_ids=start_label_ids, end_label_ids= end_label_ids)
        )
    return features

def convert_label_ids_to_onehot(label_ids, label_list):
    one_hot_labels= [[False]*len(label_list) for _ in range(len(label_ids))]
    label_map = {label: i for i, label in enumerate(label_list)}
    ignore_index= -100
    non_index= -1
    for i, label_id in enumerate(label_ids):
        for sub_label_id in label_id:
            if sub_label_id not in [ignore_index, non_index]:
                one_hot_labels[i][sub_label_id]= 1
    return one_hot_labels


