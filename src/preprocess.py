import os     
import shutil
import random #zum sortieren der bilder

# define the paths
RAW_DATA_DIR = 'data/raw' 
PROCESSED_DATA_DIR = 'data/processed'
# collect the datapath
cat_files = [os.path.join(RAW_DATA_DIR, 'cats', f) for f in os.listdir(os.path.join(RAW_DATA_DIR, 'cats'))]
dog_files = [os.path.join(RAW_DATA_DIR, 'dogs', f) for f in os.listdir(os.path.join(RAW_DATA_DIR, 'dogs'))]

