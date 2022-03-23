import pytest
import unittest

from alignments import Alignments
from translators import Translations
from utils import get_and_preprocess_dataset


@pytest.mark.alignments
class TestAlignment(unittest.TestCase):
    df1 = get_and_preprocess_dataset(wsi_task="wsi_2010")
    df2 = get_and_preprocess_dataset(wsi_task="wsi_2013")
    align = Alignments(device="cpu")
    
    def test1(self):
        text = self.df1.text.to_list()[0]
        target_word = self.df1.target_word.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translated_text = Translations.get_official_google_translate(text, source_lang, target_lang)
        aligned_words = self.align.alignment_bert(text, translated_text, target_word)
        return self.assertTrue(len(aligned_words) > 0)
    
    def test2(self):
        text = self.df1.text.to_list()[0]
        target_word = self.df1.target_word.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translated_text = Translations.get_official_google_translate(text, source_lang, target_lang)
        aligned_words = self.align.alignment_awesome(text, translated_text, target_word)
        return self.assertTrue(len(aligned_words) > 0)
    
    def test3(self):
        text = self.df1.text.to_list()[0]
        target_word = self.df1.target_word.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translated_text = Translations.get_official_google_translate(text, source_lang, target_lang)
        aligned_words = self.align.alignment_simalign(text, translated_text, target_word)
        return self.assertTrue(len(aligned_words) > 0)
