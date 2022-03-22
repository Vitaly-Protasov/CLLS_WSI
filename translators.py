from typing import Callable
from enum import Enum
import requests, uuid
import torch
import ast
from easynmt import EasyNMT
from itranslate import itranslate

from config import wsi_2010_path, wsi_2013_path


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
) -> Callable:
    return itranslate(text, from_lang=source_lang, to_lang=target_lang)


def get_microsoft_translate(
    text: str,
    source_lang: Enum,
    target_lang: Enum
) -> Callable:
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

def get_and_preprocess_dataset(self):
    if self.wsi_task == "wsi_2010":
        path = wsi_2010_path
    elif self.wsi_task == "wsi_2013":
        path = wsi_2013_path
    
    df = pd.read_csv(path)
    df['text'] = df['sentence'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    df['target_word'] = [ast.literal_eval(x)[int(i)] for i, x in zip(df.target_id, df.sentence)]
    return df
