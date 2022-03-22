from config import wsi_2010_path, wsi_2013_path
import pandas as pd
import ast


def get_and_preprocess_dataset(wsi_task):
    if wsi_task == "wsi_2010":
        path = wsi_2010_path
    elif wsi_task == "wsi_2013":
        path = wsi_2013_path
    
    df = pd.read_csv(path)
    df['text'] = df['sentence'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    df['target_word'] = [ast.literal_eval(x)[int(i)] for i, x in zip(df.target_id, df.sentence)]
    return df
