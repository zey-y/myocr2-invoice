import os
import hashlib
# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)
# project_root = os.path.dirname(current_dir)
# os.chdir(project_root)
import cv2
import json
import numpy as np
# import predict2
import random
# import predict2
import math

from main import img_joint
from PIL import Image, ImageEnhance
from main import preprocess_image2
from main import preprocess_image
from main import convert_coordinates
from main import convert_coordinates2
from paddleocr import PaddleOCR
# from main import ocr
from main import __get_img__

ocr = PaddleOCR(
  rec=r'models/ch_PP-OCRv4_rec_infer',
  rec_model_dir=r'models/ch_PP-OCRv4_rec_infer',
  det=r'models/ch_PP-OCRv4_det_infer',
  cls=r'models/ch_ppocr_mobile_v2.0_cls_infer',
  cls_model_dir=r'models/ch_ppocr_mobile_v2.0_cls_infer',
  det_model_dir=r'models/ch_PP-OCRv4_det_infer')

def generate_pretty_color():
  # 避免过亮或过暗的颜色，这里我们设置RGB每个分量的范围为[64, 191]
  min_val = 64
  max_val = 191

  # 确保颜色有一定的饱和度，通过保持RGB值之间的差异
  r = random.randint(min_val, max_val)
  g = random.randint(min_val, max_val)
  b = random.randint(min_val, max_val)

  # 增加一些变化，以确保颜色不是单调的
  if max(r, g, b) - min(r, g, b) < 64:  # 如果RGB值之间的差异小于64，则调整其中一个值
    which_to_adjust = random.choice(['r', 'g', 'b'])
    adjustment = random.choice([-64, 64])  # 随机决定是增加还是减少
    if which_to_adjust == 'r':
      r = max(min(r + adjustment, max_val), min_val)
    elif which_to_adjust == 'g':
      g = max(min(g + adjustment, max_val), min_val)
    else:
      b = max(min(b + adjustment, max_val), min_val)

  return (r, g, b)

def label(public_info):
  # 标注文件地址
  target_path = public_info.target_path
  files = os.listdir(target_path)

  label_file_name = public_info.target_path + "/Label.txt"
  label_file_ocr_name = public_info.target_ocr_path + "/Label.txt"
  file_state_name = public_info.target_path + "/fileState.txt"
  existing_lines = {}
  existing_ocr_lines = {}
  try:
    file = open(label_file_name, 'r+', encoding='utf-8')

    for readline in file.readlines():
      n = readline.split('\t')[0]
      existing_lines[n.rsplit('/')[1].split('.')[0]] = json.loads(readline.replace(n, '', 1))

  except FileNotFoundError:
    file = open(label_file_name, 'w', encoding='utf-8')
  try:
    file_ocr = open(label_file_ocr_name, 'r+', encoding='utf-8')
    for readline in file_ocr.readlines():
      n = readline.split('\t')[0]
      existing_ocr_lines[n.rsplit('/')[1].split('.')[0]] = json.loads(readline.replace(n, '', 1))
  except FileNotFoundError:
    file_ocr = open(label_file_ocr_name, 'w', encoding='utf-8')
  try:
    file_state = open(file_state_name, 'r+', encoding='utf-8')
  except FileNotFoundError:
    file_state = open(file_state_name, 'w', encoding='utf-8')
  print(existing_lines)


  for file_name in files:
    if not (file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg')):
      continue
    name = file_name.split('.')[0]
    # 生成新图片名字前缀.
    flag = "z_640_2_"
    if flag + name in existing_lines or name.startswith('z_') or name not in existing_lines:
      continue

    vs = existing_lines[name]
    xy_info = []
    for v in vs:
      points = v['points']
      left, top = points[0]
      right, bottom = points[2]

      xy_info.append([left, top, right, bottom, v['transcription']])
    file_path = os.path.join(target_path, file_name)
    imgs = __get_img__(file_name, file_path)
    for img in imgs:
      img = np.array(img)
      processed_img, orig_size, new_size, paste_coords, canvas = preprocess_image2(img, 640, 640)
      enhancer = ImageEnhance.Contrast(canvas)
      # 增强对比度
      image_enhanced = enhancer.enhance(3.0)
      pil_image = image_enhanced.convert('L')
      img_np = np.array(pil_image)
      # 可以使用以下处理生成多张标注的图片, 需要时依次取消注释.
      canvas = pil_image
      # canvas = image_enhanced

      img_file = public_info.target_path + "/" + flag + f"{name}" +".png"
      label_img_url = public_info.label_img_url + "/" + flag + f"{name}" +".png"
      canvas.save(img_file)
      converted_detections = xy_info
      canvas = np.array(canvas)
      data_line = []
      labels = {}
      for obj in converted_detections:
        left, top, right, bottom = obj[0], obj[1], obj[2], obj[3]
        label = str(obj[4])
        # confidence = obj[5]
        if label in labels:
          continue
        color = generate_pretty_color()
        labels[label] = label
        # cv2.rectangle(canvas, (left, top), (right, bottom), color=color, thickness=1, lineType=cv2.LINE_AA)
        caption = f"{label} "
        # box = convert_coordinates([left, top, right, bottom], orig_size, new_size, paste_coords)
        box = convert_coordinates2([left, top, right, bottom], orig_size, new_size, paste_coords)
        # box = convert_coordinates([left, top, right, bottom], new_size, orig_size, paste_coords)
        # o_left, o_top, o_right, o_bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        o_left, o_top, o_right, o_bottom = box[0], box[1], box[2], box[3]

        points = [[o_left, o_top], [o_right, o_top], [o_right, o_bottom], [o_left, o_bottom]]
        data = {
          "transcription": label,
          "points": points,
          "difficult": False
        }
        o_left, o_top, o_right, o_bottom = int(o_left), int(o_top), int(o_right), int(o_bottom)
        w, h = cv2.getTextSize(caption, 0, 0.5, 1)[0]
        cv2.rectangle(canvas, (o_left - 1, o_top - 20), (o_left + w + 10, o_top), color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, caption, (o_left, o_top - 5), 0, 0.5, (0, 0, 0), 1, 2)
        cv2.rectangle(canvas, (o_left, o_top), (o_right, o_bottom),color=color, thickness=2, lineType=cv2.LINE_AA)
        data_line.append(data)
      json_data = json.dumps(data_line)
      # 保存对比图.
      img_joint(Image.fromarray(img), Image.fromarray(canvas), 1).save(public_info.contrast_path + "/" + f"{name}" + ".png")
      aa =label_img_url + "\t" + json_data
      print(img_file)
      if label_img_url not in existing_lines:
        # 写入标注位置信息.
        file.write(aa + '\n')
        file_state.write(label_img_url + '\t1' + '\n')
import tool.public_info as public_info
# public_info.contrast_path生成转换后的对比图
label(public_info)