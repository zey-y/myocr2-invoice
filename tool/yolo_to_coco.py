import os
import json
from PIL import Image
from sympy import false


def get_category_id(name, categories):
    for category in categories:
        if category['name'] == name:
            return category['id']
    return None

def parse_yolo_annotation(label_path, image_id, annotation_id, categories, image_width, image_height):
    annotations = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height

            xmin = x_center - width / 2
            ymin = y_center - height / 2
            category_id = class_id + 1
            # category_id = get_category_id(label_name, categories)
            # if not category_id:
            #     category_id = class_id + 1
            #     categories.append({'id': category_id, 'name': label_name})

            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [[xmin+width, ymin, xmin+width, ymin+height, xmin, ymin+height, xmin, ymin]],
                'bbox': [xmin, ymin, width, height],
                'area': width * height,
                'iscrowd': False,
                'isbbox': True
            })
            annotation_id += 1

    return annotations, annotation_id
def convert_yolo_to_coco(label_coco_info, yolo_dir, output_file, values):
    images = []
    annotations = []
    # categories = label_coco_info
    categories = [{'id': info + 1, 'name': label_coco_info[info], 'supercategory': 'item' if label_coco_info[info].startswith("item_") else ''} for info in label_coco_info]
    # categories.append({'id': category_id, 'name': label_name})

    annotation_id = 1
    image_id = 1

    # for split in ['train', 'valid']:
    # for split in ['valid']:
    for split in values:
        image_dir = os.path.join(yolo_dir, split, 'images')
        label_dir = os.path.join(yolo_dir, split, 'labels')

        # for filename in os.listdir(label_dir):
        for filename in os.listdir("out\img2"):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, filename)
                label_path = os.path.join(label_dir, filename.rsplit('.', 1)[0] + '.txt')
                try:
                  open(label_path, 'r')
                except FileNotFoundError:
                    continue
                # label_path = os.path.join(image_dir, filename)
                # image_path = os.path.join(label_dir, filename.rsplit('.', 1)[0] + '.txt')
                try:
                  img = Image.open(image_path)
                except FileNotFoundError:
                    continue
                image_width, image_height = img.size

                images.append({
                    'id': image_id,
                    'file_name': filename,
                    'width': image_width,
                    'height': image_height
                })

                new_annotations, annotation_id = parse_yolo_annotation(
                    label_path, image_id, annotation_id, categories, image_width, image_height
                )
                annotations.extend(new_annotations)

                image_id += 1
    categories.sort(key=lambda x: x['id'])
    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
def to(public_info):
    label_coco_info = public_info.label_coco_info
    yolo_dir = public_info.dataset_path
    output_file = yolo_dir + '/coco_format_test.json'
    convert_yolo_to_coco(label_coco_info, yolo_dir, output_file, ['test'])
    output_file = yolo_dir + '/coco_format_train.json'
    convert_yolo_to_coco(label_coco_info, yolo_dir, output_file, ['train'])
    output_file = yolo_dir + '/coco_format_valid.json'
    convert_yolo_to_coco(label_coco_info, yolo_dir, output_file, ['valid'])
