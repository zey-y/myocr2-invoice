import cv2
import os
import json
import numpy as np
import predict2
import random

from main import img_joint
from PIL import Image
from main import preprocess_image
from main import convert_coordinates
from main import __get_img__

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

# 自动标注
# 使用PPOCRLabel打开target_path路径进行微调
def auto_label():
  # 待标注的目标路径
  directory = "D:\idataway\发票\发票3"
  # directory = r"D:\发票\fapiao"
  files = os.listdir(directory)
  # 标注信息保存路径.
  target_path = 'img2'
  # 缩放与原图可视化图片保存路径
  contrast_path = 'flag'
  # 文件名字, 自增.
  count = 1200
  file = open(target_path + "/Label.txt", 'w', encoding='utf-8')
  for file_name in files:
    count = count + 1
    file_path = os.path.join(directory, file_name)
    img = __get_img__(file_name, file_path)
    processed_img, orig_size, new_size, paste_coords, canvas = preprocess_image(img, 640, 640)
    img_file = target_path + "/" + f"{count}" +".png"
    # 保存图片, 缩放后的.
    canvas.save(img_file)
    # cv2.imwrite(imgfile, canvas)
    converted_detections = predict2.start(canvas)
    canvas = np.array(canvas)
    # results = model(processed_img)[0]
    # boxes = results.boxes.data.tolist()
    # names = results.names
    data_line = []
    labels = {}
    for obj in converted_detections:
      left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
      label = str(obj[4])
      confidence = obj[5]
      if label in labels:
        continue
      color = generate_pretty_color()
      # labels[label] = label
      cv2.rectangle(canvas, (left, top), (right, bottom), color=color, thickness=1, lineType=cv2.LINE_AA)
      caption = f"{label} {confidence:.2f}"
      w, h = cv2.getTextSize(caption, 0, 0.5, 1)[0]
      box = convert_coordinates([left, top, right, bottom], orig_size, new_size, paste_coords)
      o_left, o_top, o_right, o_bottom = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      img = np.array(img)
      cv2.rectangle(img, (o_left - 1, o_top - 20), (o_left + w + 10, o_top), color, thickness=-1, lineType=cv2.LINE_AA)
      cv2.putText(img, caption, (o_left, o_top - 5), 0, 0.5, (0, 0, 0), 1, 2)
      cv2.rectangle(img, (o_left, o_top), (o_right, o_bottom),color=color, thickness=2, lineType=cv2.LINE_AA)
      points = [[left, top], [right, top], [right, bottom], [left, bottom]]
      data = {
        "transcription": label,
        "points": points,
        "difficult": False
      }
      data_line.append(data)
    json_data = json.dumps(data_line)
    # 保存对比图.
    img_joint(Image.fromarray(img), Image.fromarray(canvas), 1).save(contrast_path + "/" + f"{count}" + ".png")
    aa =img_file + "	" + json_data
    print(img_file)
    # 写入标注位置信息.
    file.write(aa + '\n')

auto_label()