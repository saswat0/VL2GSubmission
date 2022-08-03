# TASK 1: Split given dataset into training and validation datasets

import os, csv
import shutil
import random
from pathlib import Path

source_dir = '../charts'
dest_dir = '../dataset'

train_val_file = f'{source_dir}/train_val.csv'
refer = {0: 'train', 1: 'val'}

print(f"Copying training and validation images to {dest_dir}")
with open(train_val_file) as f:
    reader = csv.reader(f)
    next(f)
    for row in reader:
        file_name, category = row
        mode = refer[random.choices([0,1], weights=(0.8, 0.2), k=1)[0]]

        Path(f"{dest_dir}/{mode}/{category}").mkdir(parents=True, exist_ok=True)
        dest_file_path = os.path.join(dest_dir, mode, category)
        src_file_path = os.path.join(source_dir, 'train_val', file_name+'.png')
        shutil.copy(src_file_path, dest_file_path)

print(f"Copying test images to {dest_dir}")
os.system(f'cp -r {source_dir}/test {dest_dir}/')