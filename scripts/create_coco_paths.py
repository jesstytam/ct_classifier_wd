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





coco_annotation_file = '/home/jess/ct_classifier_wd/data/processed/' + dataset + '_coco.json'
    coco = json.load(open(coco_annotation_file))

    # create lists & dictionaries
    # get image classes & create the new coco object
    img2_list = []
    image_label = annotation['category_id']
    for label in labels_list:
        category_id = annotation['category_id']+1
        if category_id == label:
            image_label = label
            img_list = {'image_path': img2_path,
                        'image_label':image_label}
            img2_list.append(img_list)
        else:
            pass

    # save the coco file
    for line in img2_list:
        json.dump(line, file)
    # for line in all_labels:
    #     file.dump(line+'\n')

    print(img2_path, ' file written!')


    labels_list = {
                1: {'label': 'Red Kangaroo'},
                2: {'label': 'Kangaroo'},
                3: {'label': 'Dingo'},
                4: {'label': 'Rabbit'},
                5: {'label': 'Cat'},
                6: {'label': 'Emu'},
                7: {'label': 'Bird'},
                8: {'label': 'Pig'},
                9: {'label': 'Euro'},
                10: {'label': 'Fox'},
                11: {'label': 'Echidna'},
                12: {'label': 'Western Grey Kangaroo'},
                13: {'label': 'Small mammal'},
                14: {'label': 'Other'},
                15: {'label': 'Goat'}
            }
