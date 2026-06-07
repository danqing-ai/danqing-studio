"""
Torch-free Chinese polyphone G2P for DiffRhythm 2 (MLX / macOS).

Upstream ``g2p/g2p/chinese_model_g2p.py`` imports PyTorch for ONNX inference only.
This module is injected as ``g2p.g2p.chinese_model_g2p`` before Amphion mandarin loads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers import BertTokenizer


class PolyDataset:
    def __init__(self, words, labels, word_pad_idx=0, label_pad_idx=-1):
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

    def preprocess(self, origin_sentences, origin_labels):
        data = []
        labels = []
        sentences = []
        for line in origin_sentences:
            words = []
            word_lens = []
            for token in line:
                words.append(token)
                word_lens.append(1)
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append(((words, token_start_idxs), 0))
        for tag in origin_labels:
            labels.append(tag)
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        sentences = [x[0][0] for x in batch]
        ori_sents = [x[0][1] for x in batch]
        labels = [x[1] for x in batch]
        batch_len = len(sentences)

        max_len = max(len(s[0]) for s in sentences)
        max_label_len = 0
        batch_data = np.ones((batch_len, max_len))
        batch_label_starts = []

        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        batch_pmasks = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]
            batch_pmasks[j][:cur_tags_len] = [
                1 if item > 0 else 0 for item in labels[j]
            ]

        return [
            batch_data.astype(np.int64),
            np.asarray(batch_label_starts, dtype=np.int64),
            batch_labels.astype(np.int64),
            batch_pmasks.astype(np.int64),
            ori_sents,
        ]


class BertPolyPredict:
    """Polyphone disambiguation via ONNX (no PyTorch)."""

    def __init__(self, bert_model: str | Path, jsonr_file: str | Path, json_file: str | Path):
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_model), do_lower_case=True)
        with open(jsonr_file, encoding="utf8") as fp:
            self.pron_dict = json.load(fp)
        with open(json_file, encoding="utf8") as fp:
            self.pron_dict_id_2_pinyin = json.load(fp)
        self.num_polyphone = len(self.pron_dict)
        self.polydataset = PolyDataset

        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_path = os.path.join(str(bert_model), "poly_bert_model.onnx")
        self.session = InferenceSession(
            onnx_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.session.disable_fallback()

    def predict_process(self, txt_list: List[Any]) -> list[str]:
        word_test, label_test, _texts_test = self.get_examples_po(txt_list)
        data = self.polydataset(word_test, label_test)
        pred_tags: list[str] = []
        for idx in range(len(data)):
            batch_samples = data.collate_fn([data[idx]])
            pred_tags.extend(self._predict_batch(batch_samples))
        return pred_tags

    def _predict_batch(self, batch_samples: list[Any]) -> list[str]:
        pred_tags: list[str] = []
        batch_data, _batch_label_starts, batch_labels, batch_pmasks, _ = batch_samples
        batch_data = np.asarray(batch_data, dtype=np.int32)
        batch_pmasks = np.asarray(batch_pmasks, dtype=np.int32)
        batch_output = self.session.run(
            output_names=["outputs"],
            input_feed={"input_ids": batch_data},
        )[0]
        label_masks = batch_pmasks == 1
        for i, indices in enumerate(np.argmax(batch_output, axis=2)):
            for j, tag_idx in enumerate(indices):
                if label_masks[i][j]:
                    pred_tags.append(self.pron_dict_id_2_pinyin[str(int(tag_idx) + 1)])
        return pred_tags

    def get_examples_po(self, text_list):
        word_list = []
        label_list = []
        sentence_list = []
        for line in [text_list]:
            sentence = line[0]
            tokens = line[0]
            index = line[-1]
            front = index
            back = len(tokens) - index - 1
            labels = [0] * front + [1] + [0] * back
            words = ["[CLS]"] + [item for item in sentence]
            words = self.tokenizer.convert_tokens_to_ids(words)
            word_list.append(words)
            label_list.append(labels)
            sentence_list.append(sentence)
            assert len(labels) + 1 == len(words)
            assert len(labels) == len(sentence)
            assert len(word_list) == len(label_list)
        return word_list, label_list, text_list
