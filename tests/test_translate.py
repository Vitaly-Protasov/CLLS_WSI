import pytest
import unittest
from langdetect import detect

from translators import Translations
from utils import get_and_preprocess_dataset

@pytest.mark.translators
class TestTranslators(unittest.TestCase):
    df1 = get_and_preprocess_dataset(wsi_task="wsi_2010")
    df2 = get_and_preprocess_dataset(wsi_task="wsi_2013")
    translation_class = Translations(device="cpu")

    def test1(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = self.translation_class.get_easynmt_translate(text, source_lang, target_lang)
        return self.assertNotIn(translation, [None, ""])

    def test2(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = self.translation_class.get_easynmt_translate(text, source_lang, target_lang)
        detected_lang = detect(translation)
        return self.assertEqual(detected_lang, target_lang)

    def test3(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = Translations.get_microsoft_translate(text, source_lang, target_lang)
        return self.assertNotIn(translation, [None, ""])

    def test4(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = Translations.get_microsoft_translate(text, source_lang, target_lang)
        detected_lang = detect(translation)
        return self.assertEqual(detected_lang, target_lang)

    def test5(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = Translations.get_official_google_translate(text, source_lang, target_lang)
        return self.assertNotIn(translation, [None, ""])

    def test6(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = Translations.get_official_google_translate(text, source_lang, target_lang)
        detected_lang = detect(translation)
        return self.assertEqual(detected_lang, target_lang)

    def test7(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = Translations.get_unofficial_google_translate(text, source_lang, target_lang)
        return self.assertNotIn(translation, [None, ""])

    def test8(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = Translations.get_unofficial_google_translate(text, source_lang, target_lang)
        detected_lang = detect(translation)
        return self.assertEqual(detected_lang, target_lang)
