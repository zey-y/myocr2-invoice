import json
from PIL import Image
from pandas import isnull

import tool.public_info as public_info
# from main import convert_coordinates

# 类别ID到文字标签的映射
category_map = public_info.label_coco_info

def convert_coordinates(box, orig_size, new_size, paste_coords = (0, 0)):
  orig_w, orig_h = orig_size
  new_w, new_h = new_size
  paste_x, paste_y = paste_coords

  x1, y1, x2, y2 = box
  scale_x = orig_w / new_w
  scale_y = orig_h / new_h

  x1_new = (x1 - paste_x) * scale_x
  y1_new = (y1 - paste_y) * scale_y
  x2_new = (x2 - paste_x) * scale_x
  y2_new = (y2 - paste_y) * scale_y

  return [x1_new, y1_new, x2_new, y2_new]

# 加载COCO数据
# with open("D:\Project\python\PaddleDetection\configs\datasets\mydata\coco_format_test.json", "r") as f:
# with open("D:\Project\python\PaddleDetection\configs\datasets\mydata\coco_format_train.json", "r") as f:
# with open("D:\Project\python\PaddleDetection\configs\datasets\mydata\coco_format_valid.json", "r") as f:

with open("D:\Project\python\myocr2-invoice\out\mydataset\coco_format_train.json", "r") as f:
# with open("D:\Project\python\myocr2-invoice\out\mydataset\coco_format_valid.json", "r") as f:
  coco_data = json.load(f)
with open("Label.txt", "r", encoding='utf-8') as f2:
  lines = set(f2.readlines())
  # 去除每行末尾的换行符
  lines = {line.rstrip('\n') for line in lines}

# 按image_id分组标注
annotations_by_image = {}
for ann in coco_data["annotations"]:
  image_id = ann["image_id"]
  if image_id not in annotations_by_image:
    annotations_by_image[image_id] = []
  annotations_by_image[image_id].append(ann)

# 生成Label.txt内容
label_lines = []
for image in coco_data["images"]:
  iw = image['width']
  ih = image['height']
  image_id = image["id"]
  file_name = image["file_name"]
  if file_name == '':
    continue

  if not file_name.startswith("z_640_1_"):
  # if file_name.startswith("z_640_0_"):
    continue
  file_name = file_name.replace("z_640_1_", "")

  if image_id not in annotations_by_image:
    continue  # 跳过无标注的图像


  with Image.open(f"out/img2/{file_name}") as img:
    width, height = img.size
    # width = width/640
    # height = height/640
  if width <= 0:
    continue


  annotations = []
  for ann in annotations_by_image[image_id]:
    category_id = ann["category_id"]
    if category_id not in category_map:
      continue  # 跳过未定义的类别
    imageId = ann['image_id']
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = convert_coordinates([x, y, x + w, y + h], (width, height), (640, 640))
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    points = [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h]
    ]
    annotations.append({
      "transcription": category_map[category_id-1],
      "points": points,
      "difficult": False
    })

  # 格式化为字符串
  line = f"img2/{file_name}\t{json.dumps(annotations, ensure_ascii=False)}"
  if not lines.__contains__(line):
    label_lines.append(line)

# 保存结果
with open("Label.txt", "a", encoding="utf-8") as f:
  f.write("\n".join(label_lines))


