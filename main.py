import cv2
import re
import numpy as np
import fitz
from easyofd.ofd import OFD
import os
import base64

import predict2
import math
from paddleocr import PaddleOCR
from flask import Flask, request
from PIL import Image
from torchvision import transforms

app = Flask(__name__)
ocr = PaddleOCR(
    rec=r'models/ch_PP-OCRv4_rec_infer',
    det=r'models/ch_PP-OCRv4_det_infer')

types = ['image/png', 'image/jpg', 'image/jpeg', 'application/pdf', 'application/ofd', 'application/octet-stream']
FIXED_WIDTH = 1219

# 图片预处理
def preprocess_image(image_path, target_w=640, target_h=640):

  img = Image.fromarray(image_path)
  orig_w, orig_h = img.size
  if orig_w > orig_h:
    new_w = target_w
    new_h = int(orig_h * (new_w / orig_w))
  else:
    new_h = target_h
    new_w = int(orig_w * (new_h / orig_h))

  # Resize the image
  img_resized = img.resize((new_w, new_h))

  # Create a black canvas with the target size
  canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
  # Paste the resized image onto the canvas
  paste_x = (target_w - new_w) // 2
  paste_y = (target_h - new_h) // 2
  canvas.paste(img_resized, (paste_x, paste_y))
  transform = transforms.ToTensor()
  input_tensor = transform(canvas).unsqueeze(0)
  return input_tensor, (orig_w, orig_h), (new_w, new_h), (paste_x, paste_y), canvas

# 缩放图片到原始图片映射
def convert_coordinates(box, orig_size, new_size, paste_coords):
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

def __get_img__(filename, file):
  filename = filename.lower()
  if filename.endswith('.ofd'):
    ofd = OFD()
    if isinstance(file, str):
      with open(file, "rb") as f:
         ofdb64 = str(base64.b64encode(f.read()), "utf-8")
    else:
      ofdb64 = str(base64.b64encode(file), "utf-8")
    print(ofdb64)
    ofd.read(ofdb64, save_xml=False, xml_name=f"{os.path.split(filename)[0]}_xml")
    img_np = ofd.to_jpg()
    ofd.del_data()
    img = np.array(img_np[0])
  elif filename.endswith('.pdf'):
    if isinstance(file, str):
      doc = fitz.open(file)
    else:
      doc = fitz.open("pdf", file)
    if len(doc) == 0:
      return {}
    page = doc.load_page(0)
    original_width = page.rect.width
    scale_factor = FIXED_WIDTH / original_width
    pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
  else:
    if isinstance(file, str):
      img = cv2.imread(file)[:, :, ::-1]
    else:
      img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.COLOR_BGR2RGB)
    # processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
  return img

@app.route('/invoice_ocr', methods=['POST'])
def invoice_ocr():
  uploaded_file = request.files['file']
  print(uploaded_file.content_type)
  if uploaded_file is None or uploaded_file.content_type not in types:
    return {}
  read = uploaded_file.read()
  filename = uploaded_file.filename
  img = __get_img__(filename, read)

  processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
  ocrResult = {}
  converted_detections = predict2.start(resize_img)
  for obj in converted_detections:
    left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
    box = convert_coordinates([left, top, right, bottom], orig_size, new_size, paste_coords)
    left, top, right, bottom = box
    label = str(obj[4])
    cropped_img = img[math.floor(top):math.ceil(bottom), math.floor(left):math.ceil(right)]
    rr = ocr.ocr(cropped_img, det=False, cls=False)
    for line in rr:
      if line is None:
        continue
      for word_info in line:
        ocrResult[label] = re.sub(r'([￥¥]) *', '', word_info[0]).strip()
  return ocrResult





def img_joint(new_img, old_img, axis=0):
  w1, h1 = old_img.size
  w2, h2 = new_img.size
  max_width = max(w1, w2) + 4
  max_height = max(h1, h2) + 4
  top1 = (max_height - h1) // 2
  top2 = (max_height - h2) // 2
  color = (0, 0, 0)
  if axis == 1:
    padded_img1 = Image.new('RGB', (w1, max_height), color)
    padded_img2 = Image.new('RGB', (w2, max_height), color)

    padded_img1.paste(old_img, (0, top1))
    padded_img2.paste(new_img, (0, top2))
    im = np.array(padded_img1)
    im2 = np.array(padded_img2)
    im2 = np.concatenate((im2, im), axis=1)
    new_img = Image.fromarray(im2)
    return new_img
  else:
    padded_img1 = Image.new('RGB', (max_width, h1), color)
    padded_img2 = Image.new('RGB', (max_width, h2), color)
    padded_img1.paste(old_img, (0, 0))
    padded_img2.paste(new_img, (0, 0))
    im = np.array(padded_img1)
    im2 = np.array(padded_img2)
    im2 = np.concatenate((im2, im), axis=0)
    new_img = Image.fromarray(im2)
    return new_img

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)
