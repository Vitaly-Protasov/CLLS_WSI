from WordAlignment import WordAlignment
import transformers
import re
import itertools
import torch
from enum import Enum


class Alignments:
    def __init__(self, device: Enum, model_name: Enum = "bert-base-multilingual-cased"):
        self.model_name = model_name
        self.model1 = WordAlignment(model_name=model_name, tokenizer_name=model_name, device=device, fp16=False)
        self.model2 = transformers.BertModel.from_pretrained(model_name)
        self.tokenizer2 = transformers.BertTokenizer.from_pretrained(model_name)
    
    def _clear_word(self, word: str) -> str:
        return re.sub(r'[^\w]', '', word.lower().strip())
    
    def alignment_1(self, sent_original: str, sent_translated: str, target_w: str):
        """
        Alignment from https://github.com/andreabac3/Word-Alignment-BERT
        """
        _, decoded = self.model1.get_alignment(sent_original.split(), sent_translated.split(), calculate_decode=True)
        for sentence1_w, sentence2_w in decoded:
            sentence1_w = self._clear_word(sentence1_w)
            sentence2_w = self._clear_word(sentence2_w)
            if sentence1_w == target_w:
                start = sent_translated.find(sentence2_w)
                end = start + len(sentence2_w)
                return f'{start}-{end}', sentence2_w
        return f'{0}-{0}', ''

    def alignment_2(self, sent_original: str, sent_translated: str, target_w: str):
        """
        Alignment from https://github.com/neulab/awesome-align
        """
        sent_src, sent_tgt = sent_original.strip().split(), sent_translated.strip().split()
        token_src, token_tgt = [self.tokenizer2.tokenize(word) for word in sent_src], [self.tokenizer2.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer2.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer2.convert_tokens_to_ids(x) for x in token_tgt]
        ids_src, ids_tgt = self.tokenizer2.prepare_for_model(list(itertools.chain(*wid_src)), \
                                                             return_tensors='pt', model_max_length=self.tokenizer2.model_max_length, truncation=True)['input_ids'],\
        self.tokenizer2.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=self.tokenizer2.model_max_length)['input_ids']

        sub2word_map_src = []
        for i, word_list in enumerate(token_src):
            sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            sub2word_map_tgt += [i for x in word_list]

        # alignment
        align_layer = 8
        threshold = 1e-3
        self.model2.eval()
        with torch.no_grad():
            out_src = self.model2(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            out_tgt = self.model2(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
            
            dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)
            softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = set()
        for i, j in align_subwords:
            align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

        for i, j in align_words:
            sentence1_w = self._clear_word(sent_src[i])
            sentence2_w = self._clear_word(sent_tgt[j])
            if sentence1_w == target_w:
                start = sent_translated.find(sentence2_w)
                end = start + len(sentence2_w)
                return f'{start}-{end}', sentence2_w
        return f'{0}-{0}', ''
