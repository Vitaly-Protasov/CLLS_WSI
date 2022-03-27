from pathlib import Path


home_path = Path(__file__).resolve().parent.parent
wsi_2010_path = Path(home_path, "data/dataset_wsi_2010.csv")
wsi_2013_path = Path(home_path, "data/dataset_wsi_2013.csv")

azure_subscription_key = "<your azure code>"
translations_folder = str(Path(home_path, "translation_files"))