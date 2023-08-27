import json
from PIL import Image
import os

def crop_image_to_bbox(dataset):

    coco_annotation_file = '/home/jess/ct_classifier_wd/data/processed/' + dataset + '_coco.json'
    coco = json.load(open(coco_annotation_file))

    for annotation in coco['annotations']:

        try: 
            bbox = annotation['bbox']
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]

            box = (x, y, x+w, y+h)

            image_path = ''+ annotation['image_id'].replace('_', ' ', 4)
            image_path_updated = image_path.replace(' ', '/', 1)
            outpath = '/home/jess/data/wild_deserts/processed/crops/' + dataset + '/'

            img = Image.open(os.path.join('/home/jess/data/wild_deserts/Beyond the Fence- Tagged/images/' + image_path_updated + '.JPG'))
            img2 = img.crop(box)
            img2_path = outpath + annotation['image_id'] + '.JPG'
            img2.save(img2_path)
            
            print(img2_path, 'saved!')

            #get label
            # image_url = '/home/jess/ct_classifier_wd/data/processed/train_coco.json'
            # input_json = json.load(open(image_url))

            dict = {}

            # for cropped_image in os.listdir('/home/jess/data/wild_deserts/processed/crops/' + dataset):
            im = Image.open(img2_path)
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

        except FileNotFoundError:
            pass

    print('Cropping done for ' + dataset + ' images.')

crop_image_to_bbox('train')
crop_image_to_bbox('val')
crop_image_to_bbox('test')