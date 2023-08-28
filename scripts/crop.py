import json
from PIL import Image
import os

def crop_image_to_bbox(dataset):

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

    coco_annotation_file = '/home/jess/ct_classifier_wd/data/processed/' + dataset + '_coco.json'
    coco = json.load(open(coco_annotation_file))

    for annotation in coco['annotations']:

        for species in labels_list:
            crops_path = '/home/jess/data/wild_deserts/processed/crops'
            sp = str(labels_list[species].values()).split("'")[1]
            save_path = os.path.join(crops_path, sp)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if annotation['category_id']==species:

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

            print('Images for ', sp, ' saved!')

        print(dataset, ' images processing done!')

crop_image_to_bbox('train')
crop_image_to_bbox('val')
crop_image_to_bbox('test')