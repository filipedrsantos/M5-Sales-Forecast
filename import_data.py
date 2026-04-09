import os
import kaggle
import zipfile

kaggle.api.authenticate()

file = kaggle.api.competition_list_files('m5-forecasting-accuracy')

for f in file.files:
    print(f.name)

os.makedirs('data_raw', exist_ok=True)

kaggle.api.competition_download_files(competition='m5-forecasting-accuracy', path='data_raw')

with zipfile.ZipFile('data_raw/m5-forecasting-accuracy.zip', 'r') as zip_ref:
    zip_ref.extractall('data_raw')