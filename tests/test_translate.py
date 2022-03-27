import pytest
import unittest
from langdetect import detect

from nmt_wsi.translations.translators import ClassTranslations
from nmt_wsi.utils import get_and_preprocess_dataset, clear_word
from nmt_wsi.config import azure_subscription_key


@pytest.mark.translators
class TestTranslators(unittest.TestCase):
    df1 = get_and_preprocess_dataset(wsi_task="wsi_2010")
    df2 = get_and_preprocess_dataset(wsi_task="wsi_2013")
    translation_class = ClassTranslations(device="cpu")

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
        if azure_subscription_key != "<your azure code>":
            translation = ClassTranslations.get_microsoft_translate(text, source_lang, target_lang)
            return self.assertNotIn(translation, [None, ""])
        return True

    def test4(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        if azure_subscription_key != "<your azure code>":
            translation = ClassTranslations.get_microsoft_translate(text, source_lang, target_lang)
            detected_lang = detect(translation)
            return self.assertEqual(detected_lang, target_lang)
        return True

    def test5(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = ClassTranslations.get_official_google_translate(text, source_lang, target_lang)
        return self.assertNotIn(translation, [None, ""])

    def test6(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = ClassTranslations.get_official_google_translate(text, source_lang, target_lang)
        detected_lang = detect(translation)
        return self.assertEqual(detected_lang, target_lang)

    def test7(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = ClassTranslations.get_unofficial_google_translate(text, source_lang, target_lang)
        return self.assertNotIn(translation, [None, ""])

    def test8(self):
        text = self.df1.text.to_list()[0]
        source_lang = "en"
        target_lang = "ru"
        translation = ClassTranslations.get_unofficial_google_translate(text, source_lang, target_lang)
        detected_lang = detect(translation)
        return self.assertEqual(detected_lang, target_lang)

    def test9(self):
        text = "dog"
        source_lang = "en"
        target_lang = "ru"
        translation = self.translation_class.get_easynmt_translate(text, source_lang, target_lang)
        return self.assertEqual(clear_word(translation), "собака")

    def test10(self):
        text = "dog"
        source_lang = "en"
        target_lang = "ru"
        if azure_subscription_key != "<your azure code>":
            translation = self.translation_class.get_microsoft_translate(text, source_lang, target_lang)
            return self.assertEqual(clear_word(translation), "собака")
        return True

    def test11(self):
        text = "dog"
        source_lang = "en"
        target_lang = "ru"
        translation = self.translation_class.get_official_google_translate(text, source_lang, target_lang)
        return self.assertEqual(clear_word(translation), "собака")

    def test12(self):
        text = "dog"
        source_lang = "en"
        target_lang = "ru"
        translation = self.translation_class.get_unofficial_google_translate(text, source_lang, target_lang)
        return self.assertEqual(clear_word(translation), "собака")