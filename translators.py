from typing import Callable
from enum import Enum
import requests, uuid
import torch
import ast
from easynmt import EasyNMT
import pandas as pd
from itranslate import itranslate
from Naked.toolshed.shell import muterun_js


def get_easynmt_model(
    model_name: Enum = "opus-mt",
    device: Enum = None
) -> EasyNMT:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EasyNMT(model_name, device=device)
    return model


def get_unofficial_google_translate(
    text: str,
    source_lang: Enum,
    target_lang: Enum
) -> str:
    return itranslate(text, from_lang=source_lang, to_lang=target_lang)


def get_microsoft_translate(
    text: str,
    source_lang: Enum,
    target_lang: Enum
) -> str:
    subscription_key = "857b59237e77405cb60dbe0e1dfe46e7"
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
    return response.stdout.decode("utf-8")
