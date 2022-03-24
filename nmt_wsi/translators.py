from typing import Union, List, Optional
from enum import Enum
import requests, uuid
import torch
import ast
from easynmt import EasyNMT
import os
from itranslate import itranslate
from Naked.toolshed.shell import muterun_js

from nmt_wsi import config


class Translations:
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
        return self.easynmt.translate(texts, target_lang=target_lang, source_lang=source_lang)

    @staticmethod
    def get_unofficial_google_translate(
        text: str,
        source_lang: Enum,
        target_lang: Enum
    ) -> str:
        return itranslate(text, from_lang=source_lang, to_lang=target_lang)

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
        return ast.literal_eval(requested_text)[0]['translations'][0]['text']

    @staticmethod
    def get_official_google_translate(
        text: str,
        source_lang: Enum,
        target_lang: Enum
    ) -> str:
        file_translation = "trans.js"
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
