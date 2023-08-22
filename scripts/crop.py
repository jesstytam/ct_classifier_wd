import json
from PIL import Image

def crop_image_to_bbox(dataset):

    coco_annotation_file = '/home/jess2/ct_classifier_wd/data/processed/' + dataset + '_coco.json'
    coco = json.load(open(coco_annotation_file))

    for annotation in coco['annotations']:

        x = annotation['bbox'][0]
        y = annotation['bbox'][1]
        w = annotation['bbox'][2]
        h = annotation['bbox'][3]

        box = (x, y, x+w, y+h)

        image_path = ''+ annotation['image_id'].replace('_', ' ')
        image_path_updated = image_path.replace(' ', '/', 1)
        outpath = '/home/jess2/data/wild_deserts/processed/crops'

        img = Image.open(image_path_updated)
        img2 = img.crop(box)
        img2.save(outpath)

    print('Cropping done for ' + dataset ' images.')

crop_image_to_bbox('train')
crop_image_to_bbox('val')
crop_image_to_bbox('test')