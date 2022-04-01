import multiprocessing
from multiprocessing import Pool
from enum import Enum
from typing import Callable, Optional, List
import numpy as np
from tqdm import tqdm

from nmt_wsi.translations.translators import ClassTranslations


class PipelineTranslation:
    def __init__(self, device: Optional[Enum]) -> None:
        self.device = device
        self.translation_class = ClassTranslations(device=device)

    def worker(
        self,
        model_nmt: Callable,
        text: str,
        source_lang: Enum,
        target_lang: Enum
    ) -> str:
        translation = model_nmt(text, source_lang, target_lang)
        return translation

    def get_nmt_model(self, translation_model_name: Enum) -> Callable:
        if translation_model_name == "off_google":
            model_nmt = self.translation_class.get_official_google_translate
        elif translation_model_name == "unoff_google":
            model_nmt = self.translation_class.get_unofficial_google_translate
        elif translation_model_name == "microsoft":
            model_nmt = self.translation_class.get_microsoft_translate
        elif translation_model_name == "easynmt":
            model_nmt = self.translation_class.get_easynmt_translate
        else:
            raise f"Unknown translation model {translation_model_name}"
        return model_nmt

    def get_correct_cores(self, cores: int, translation_model_name: Enum) -> int:
        if cores == -1:
            cores = multiprocessing.cpu_count()
        if translation_model_name == "unoff_google":
            cores = 1
        return cores

    def paralled_translation(
        self,
        translation_model_name: Enum,
        original_sents: List[str],
        source_lang: Enum,
        target_lang: Enum,
        cores: int = -1
    ) -> List[str]:
        translations = []
        model_nmt = self.get_nmt_model(translation_model_name)
        if translation_model_name == "easynmt":
            results = model_nmt(original_sents, source_lang, target_lang)
            translations += results
            return translations
        else:
            cores = self.get_correct_cores(cores, translation_model_name)
            for i in tqdm(range(0, len(original_sents), cores)):
                sentences = original_sents[i:i+cores]
                source_langs = np.repeat(source_lang, cores)
                target_langs = np.repeat(target_lang, cores)
                models = np.repeat(model_nmt, cores)
                args = [
                    [m, s, l1, l2] for m, s, l1, l2 in zip(
                    models, sentences, source_langs, target_langs
                    )]
                with Pool(cores) as pool:
                    result = pool.starmap(self.worker, args)
                translations += result
        return translations
