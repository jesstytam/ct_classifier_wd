from pycocotools.coco import COCO
from PIL import Image
import os

def get_crop(dataset):
    
    # Path to your COCO annotations file and image directory
    coco_annotation_file = '/home/jess2/ct_classifier_wd/data/processed/' + dataset + '_coco.json'
    image_directory = '/home/jess2/data/wild_deserts/Beyond the Fence- Tagged/images'

    # get species list
    species = []
    for species in coco_train['categories']:
        category = speices['name']
        species.append(category)

    # Initialize COCO instance
    coco = COCO(coco_annotation_file)

    # Get all category IDs you're interested in (e.g., person, car, etc.)
    for id in species:
        category_ids = coco.getCatIds(catNms=[id])  # Replace 'person' with your desired category

    # Loop through images and annotations
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=category_ids)
        anns = coco.loadAnns(ann_ids)
        
        # Load the image
        image_path = os.path.join(image_directory, img_info['file_name'])
        image = Image.open(image_path)

        # Crop and save individual objects based on annotations
        for ann in anns:
            bbox = ann['bbox']
            x, y, width, height = map(int, bbox)
            object_image = image.crop((x, y, x + width, y + height))
            
            # Save the cropped object image
            save_path = f'/home/jess2/data/wild_deserts/processed/crops/train/{img_info["file_name"]}_object_{ann["id"]}.jpg'
            object_image.save(save_path)

    print("Cropping completed.")
