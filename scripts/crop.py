import json
from PIL import Image
import os
from pathlib import Path

def crop_image_to_bbox():

    labels_list = {
        0: {'label': 'Red Kangaroo'},
        1: {'label': 'Kangaroo'},
        2: {'label': 'Dingo'},
        3: {'label': 'Rabbit'},
        4: {'label': 'Cat'},
        5: {'label': 'Emu'},
        6: {'label': 'Bird'},
        7: {'label': 'Pig'},
        8: {'label': 'Euro'},
        9: {'label': 'Fox'},
        10: {'label': 'Echidna'},
        11: {'label': 'Western Grey Kangaroo'},
        12: {'label': 'Small mammal'},
        13: {'label': 'Other'},
        14: {'label': 'Goat'}
    }

    coco_annotation_file = '/home/jess/ct_classifier_wd/data/intermediate/coco.json'
    coco = json.load(open(coco_annotation_file))

    cls_coco = {
        "images": [],
        "categories": [],
        "annotations": []
    }

    categories = {
        'id': 0, 'name': 'Red Kangaroo',
        'id': 1, 'name': 'Kangaroo',
        'id': 2, 'name': 'Dingo',
        'id': 3, 'name': 'Rabbit',
        'id': 4, 'name': 'Cat',
        'id': 5, 'name': 'Emu',
        'id': 6, 'name': 'Bird',
        'id': 7, 'name': 'Pig',
        'id': 8, 'name': 'Euro',
        'id': 9, 'name': 'Fox',
        'id': 10, 'name': 'Echidna',
        'id': 11, 'name': 'Western Grey Kangaroo',
        'id': 12, 'name': 'Small mammal',
        'id': 13, 'name': 'Other',
        'id': 14, 'name': 'Goat'
    }
    
    cls_coco['categories'].append(categories)

    for annotation in coco['annotations']:

        for species in labels_list:
            crops_path = '/home/jess/data/wild_deserts/processed/crops'
            sp = str(labels_list[species].values()).split("'")[1]
            save_path = os.path.join(crops_path, sp)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if annotation['category_id']==species:

                #GET IMAGE CROP
                try: 

                    # crop the images
                    bbox = annotation['bbox']
                    x = bbox[0]
                    y = bbox[1]
                    w = bbox[2]
                    h = bbox[3]

                    box = (x, y, x+w, y+h)

                    image_path = annotation['image_id'].replace('_', ' ', 4)
                    image_path_updated = image_path.replace(' ', '/', 1)

                    img = Image.open(os.path.join('/home/jess/data/wild_deserts/Beyond the Fence- Tagged/images/' + image_path_updated + '.JPG'))
                    img2 = img.crop(box)

                    # Define a counter for filename uniqueness
                    counter = 0

                    # Generate a filename and check for uniqueness
                    while True:
                        img2_path = os.path.join(save_path, f"{annotation['image_id']}_{counter}.JPG")
                        if not os.path.exists(img2_path):
                            break
                        counter += 1
                    
                    img2.save(img2_path)
                    print(img2_path, 'saved!')

                except FileNotFoundError:
                    pass

                #GET ANNOTATION
                im = Image.open(img2_path)
                width, height = im.size

                img = {
                    "file_name": f'{sp}/{Path(img2_path).name}',
                    "height": height,
                    "width": width,
                    "id": annotation['image_id']
                }
                cls_coco['images'].append(img)

                an = {
                    "id": annotation['id'],
                    "image_id": annotation['image_id'],
                    "category_id": annotation['category_id'],
                    "iscrowd": 0,
                    "segmentation": [],
                    "area": width * height
                }
                cls_coco["annotations"].append(an)

                cls_coco_path = '/home/jess/ct_classifier_wd/data/intermediate/cls_coco.json'
                with open(cls_coco_path, 'w') as file:
                    json.dump(cls_coco, file)
                    print(cls_coco)

crop_image_to_bbox()