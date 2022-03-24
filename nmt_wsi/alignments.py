import transformers
import itertools
import torch
from enum import Enum
from typing import Optional, Tuple, List
from simalign import SentenceAligner

from nmt_wsi.WordAlignment import WordAlignment
from nmt_wsi.utils import clear_word


class Alignments:
    def __init__(
        self,
        model_name: Enum = "bert-base-multilingual-cased",
        device: Optional[Enum]=None
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model_name = model_name

        self.model1 = WordAlignment(model_name=model_name, tokenizer_name=model_name, device=self.device, fp16=False)
        self.model2 = transformers.BertModel.from_pretrained(model_name)
        self.tokenizer2 = transformers.BertTokenizer.from_pretrained(model_name)
        self.model3 = SentenceAligner(model="xlmr", token_type="word", device=device)
    
    def alignment_bert(
        self,
        sent_original: str,
        sent_translated: str,
        target_w: str
    ) -> List[Tuple[str, str]]:
        """
        Alignment from https://github.com/andreabac3/Word-Alignment-BERT
        """
        _, decoded = self.model1.get_alignment(sent_original.split(), sent_translated.split(), calculate_decode=True)

        possible_translations = []
        start = 0
        for sentence1_w, sentence2_w in decoded:
            sentence1_w = clear_word(sentence1_w)
            sentence2_w = clear_word(sentence2_w)
            if sentence1_w == target_w:
                start_tw = sent_translated[start:].find(sentence2_w) + start 
                end_tw = start_tw + len(sentence2_w)
                possible_translations.append((f'{start_tw}-{end_tw}', sentence2_w))
            start += len(sentence1_w)
        return possible_translations

    def alignment_awesome(
        self,
        sent_original: str,
        sent_translated: str,
        target_w: str
    ) -> Tuple[str, str]:
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

        possible_translations = []
        for i, j in align_words:
            sentence1_w = clear_word(sent_src[i])
            sentence2_w = clear_word(sent_tgt[j])
            if sentence1_w == target_w:
                start_tw = sent_translated.find(sentence2_w)
                end_tw = start_tw + len(sentence2_w)
                possible_translations.append((f'{start_tw}-{end_tw}', sentence2_w))
        return possible_translations
    
    def alignment_simalign(
        self,
        sent_original: str,
        sent_translated: str,
        target_w: str
    ) -> Tuple[str, str]:
        """
        Alignment from https://github.com/cisnlp/simalign
        """
        src_sentence = sent_original.split()
        trg_sentence = sent_translated.split()

        alignments = self.model3.get_word_aligns(src_sentence, trg_sentence)
        decoded = alignments['mwmf']

        possible_translations = []
        start = 0
        for sentence1_w_id, sentence2_w_id in decoded:
            sentence1_w = clear_word(src_sentence[sentence1_w_id])
            sentence2_w = clear_word(trg_sentence[sentence2_w_id])
            if sentence1_w == target_w:
                start_tw = sent_translated[start:].find(sentence2_w) + start 
                end_tw = start_tw + len(sentence2_w)
                possible_translations.append((f'{start_tw}-{end_tw}', sentence2_w))
            start += len(sentence1_w)
        return possible_translations
