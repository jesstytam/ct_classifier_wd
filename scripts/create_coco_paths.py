import json
import os
from PIL import Image

image_url = '/home/jess/ct_classifier_wd/data/processed/train_coco.json'
input_json = json.load(open(image_url))

dict = {}

for cropped_image in os.listdir('/home/jess/data/wild_deserts/processed/crops/' + dataset):

    cropped_image_path = os.path.abspath(cropped_image)
    im = Image.open(cropped_image_path)
    width, height = im.size

    # initialise dictionary
    new_dict = {
        "image_name":cropped_image_path,
        "image_details":{
            "format":"jpg",
            "width":width,
            "height":height
        },
        "label":"class_name",
    }