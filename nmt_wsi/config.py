from pathlib import Path


home_path = Path(__file__).resolve().parent.parent
wsi_2010_path = Path(home_path, 'data/dataset_wsi_2010.csv')
wsi_2013_path = Path(home_path, 'data/dataset_wsi_2013.csv')

azure_subscription_key = "857b59237e77405cb60dbe0e1dfe46e7"
translations_folder = "translations"