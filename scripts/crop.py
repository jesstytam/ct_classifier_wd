import json
from PIL import Image
import os

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

def crop_image_to_bbox(dataset):

    coco_annotation_file = '/home/jess/ct_classifier_wd/data/processed/' + dataset + '_coco.json'
    coco = json.load(open(coco_annotation_file))

    for annotation in coco['annotations']:

        try: 

            # crop the images
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

            # create lists & dictionaries
            # label list for the second part of the json
            all_labels = []
            for item in labels_list:
                species = item['label']
                all_labels.append(species)
            labels = {'labels': [all_labels]}
            
            # get image classes & create the new coco object
            img2_list = []
            image_label = annotation['category_id']
            for label in labels_list:
                category_id = annotation['category_id']+1
                if category_id == label.keys():
                    image_label = label
                    img_list = {'image_path': img2_path,
                                'image_label':image_label}
                    img2_list.append(img_list)
                else:
                    pass

            # save the coco file
            coco_outpath = '/home/jess/ct_classifier_wd/data/processed/' + dataset + '_classifier.json'
            with open(coco_outpath, 'w') as file:
                for line in img2_list:
                    file.dump(line+'\n')
                for line in all_labels:
                    file.dump(line+'\n')

        except FileNotFoundError:
            pass

    print('Cropping done for ' + dataset + ' images.')

crop_image_to_bbox('train')
# crop_image_to_bbox('val')
# crop_image_to_bbox('test')