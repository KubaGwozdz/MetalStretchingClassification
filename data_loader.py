import pandas as pd
import os
import re
from PIL import Image
import numpy as np

DATA_DIR = './data'
RAW_DATA_DIR = './rawData'
LOW_RM_DIR = '/lowRm'
HIGH_RM_DIR = '/highRm'
UNKNWN_MID_RM_DIR = '/unknownMidRm'

TARGET_ZOOM = 500

TARGET_SIZE = (335, 251)


# We treat HIGH as 1 and low as 0


def load_from_dir(dir, dir_cat, data):
    for file in os.listdir(dir):
        if file.endswith(".jpg"):
            filename = os.path.join(dir, file)
            category = dir_cat
            zoom = int(re.search('\d+x', file).group()[:-1])

            data['filename'].append(filename)
            data['category'].append(category)
            data['zoom'].append(zoom)


def load_data(data_dir):
    data = {
        'filename': [],
        'category': [],
        'zoom': [],
    }
    load_from_dir(data_dir + LOW_RM_DIR, 0, data)
    load_from_dir(data_dir + HIGH_RM_DIR, 1, data)
    df = pd.DataFrame(data)
    return df


def resize_images(df):
    for index, data in df.iterrows():
        original = Image.open(data.filename)
        if data.zoom == TARGET_ZOOM:
            to_save = original.resize(TARGET_SIZE)
            if data.category == 1:
                to_save.save(DATA_DIR + HIGH_RM_DIR + f'/{index}_original_500x.jpg')
            else:
                to_save.save(DATA_DIR + LOW_RM_DIR + f'/{index}_original_500x.jpg')
        else:
            zoom_factor = int(TARGET_ZOOM // data.zoom)
            (org_w, org_h) = original.size
            target_w = org_w // zoom_factor
            target_h = org_h // zoom_factor
            for row in range(zoom_factor):
                for col in range(zoom_factor):
                    left = row * target_w
                    top = col * target_h
                    right = (row + 1) * target_w
                    bottom = (col + 1) * target_h
                    cropped = original.crop((left, top, right, bottom))
                    colors = cropped.histogram()[230:256]
                    if not (sum(colors) > 3000 and bottom > (zoom_factor - 1) * target_h):
                        to_save = cropped.resize(TARGET_SIZE)
                        if data.category == 1:
                            to_save.save(DATA_DIR + HIGH_RM_DIR + f'/{index}_{row}_{col}_500x.jpg')
                        else:
                            to_save.save(DATA_DIR + LOW_RM_DIR + f'/{index}_{row}_{col}_500x.jpg')


