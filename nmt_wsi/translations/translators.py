from typing import Union, List, Optional
from enum import Enum
import requests, uuid
import torch
import ast
from easynmt import EasyNMT
import os
from googletrans import Translator
from Naked.toolshed.shell import muterun_js
import uuid
from pathlib import Path

from nmt_wsi import config


os.makedirs(config.translations_folder, exist_ok=True)


class UnknownLanguage(Exception):
    pass

class MicrosoftTranslationError(Exception):
    pass


class ClassTranslations:
    def __init__(self, easy_nmt_model: Enum="opus-mt", device: Optional[Enum]=None):
        self.easynmt = self.get_easynmt_model(easy_nmt_model, device)
    
    def get_easynmt_model(
        self,
        model_name: Enum = "opus-mt",
        device: Optional[Enum] = None
    ) -> EasyNMT:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EasyNMT(model_name, device=device)
        return model

    def get_easynmt_translate(
        self,
        texts: Union[str, List[str]],
        source_lang: Enum,
        target_lang: Enum
    ):
        try:
            return self.easynmt.translate(texts, target_lang=target_lang, source_lang=source_lang)
        except Exception as e:
            error_text = str(e)
            if error_text.startswith("404 Client Error"):
                raise UnknownLanguage(f"Languages \"{target_lang}\" or \"{source_lang}\" are unknown.")

    @staticmethod
    def get_unofficial_google_translate(
        text: str,
        source_lang: Enum,
        target_lang: Enum
    ) -> str:
        """
        We advise not to translate using this function in multiprocessing manner.
        """
        return Translator().translate(text, src=source_lang, dest=target_lang).text

    @staticmethod
    def get_microsoft_translate(
        text: str,
        source_lang: Enum,
        target_lang: Enum
    ) -> str:
        subscription_key = config.azure_subscription_key
        endpoint = "https://api.cognitive.microsofttranslator.com"
        location = "westeurope"
        path = '/translate'
        constructed_url = endpoint + path
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': target_lang
        }
        headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
        }
        body = [{
            'text': text
        }]
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        requested_text = request.text
        try:
            return ast.literal_eval(requested_text)[0]['translations'][0]['text']
        except:
            raise MicrosoftTranslationError(f"Something went wrong in translation from \"{source_lang}\" to \"{target_lang}\".")

    @staticmethod
    def get_official_google_translate(
        text: str,
        source_lang: Enum,
        target_lang: Enum
    ) -> str:
        file_translation = str(Path(config.translations_folder, f"{uuid.uuid4().hex}.js"))
        text = text.replace('\n', ' ').replace('\'', '\"')

        template = f"""const translate = require('@iamtraction/google-translate');
        translate(
            '{text}',
            {{from: '{source_lang}', to: '{target_lang}' }}).then(res => {{
        console.log(res.text); }}).catch(err => {{
        console.error(err);
        }});
        """
        with open(file_translation, "w", encoding="utf-8") as f:
            f.write(template)
        response = muterun_js(file_translation)

        os.remove(file_translation)
        return response.stdout.decode("utf-8")
