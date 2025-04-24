import cv2
import re
import numpy as np
import fitz
from easyofd.ofd import OFD
import os
import base64
import logging
import time

import predict2
import math
from paddleocr import PaddleOCR
from flask import Flask, request
from PIL import Image, ImageEnhance
from torchvision import transforms
import concurrent.futures
logging.disable(logging.DEBUG)
app = Flask(__name__)
app.json.ensure_ascii = False
ocr = PaddleOCR(rec=r'models/ch_PP-OCRv4_rec_infer',
                rec_model_dir=r'models/ch_PP-OCRv4_rec_infer',
                det=r'models/ch_PP-OCRv4_det_infer',
                cls=r'models/ch_ppocr_mobile_v2.0_cls_infer',
                cls_model_dir=r'models/ch_ppocr_mobile_v2.0_cls_infer',
                det_model_dir=r'models/ch_PP-OCRv4_det_infer')
# ocr_en = PaddleOCR(rec=r'models/en_PP-OCRv4_rec_infer')
# ch_class_list = ["title", "issue_date", "buyer_name", "seller_name", "invoice_clerk"]
# 多行识别的
det_true_list = set(["password_area", "buyer_address_telephone", "buyer_bank_account",
                 "seller_address_telephone", "seller_bank_account", "remark", "competent_tax_authorities_and_code"])
types = set(['image/png', 'image/jpg', 'image/jpeg', 'application/pdf',
         'application/ofd', 'application/octet-stream'])
# FIXED_WIDTH = 1219
FIXED_WIDTH = 2500


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

def preprocess_image2(img, target_w=512, target_h=512):
  if not isinstance(img, Image.Image):
    img = Image.fromarray(img)

  orig_w, orig_h = img.size
  img_resized = img.resize((target_w, target_h))
  transform = transforms.ToTensor()
  input_tensor = transform(img_resized).unsqueeze(0)
  return input_tensor, (orig_w, orig_h),img_resized.size, (0, 0), img_resized


# 缩放图片到原始图片映射
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

def convert_coordinates2(box, orig_size, new_size, paste_coords):
  orig_w, orig_h = orig_size
  new_w, new_h = new_size
  paste_x, paste_y = paste_coords

  x1, y1, x2, y2 = box
  scale_x =  new_w / orig_w
  scale_y =  new_h / orig_h

  x1_new = x1 * scale_x + paste_x
  y1_new = y1 * scale_y + paste_y
  x2_new = x2 * scale_x + paste_x
  y2_new = y2 * scale_y + paste_y

  return [x1_new, y1_new, x2_new, y2_new]


def __get_img__(filename, file):
  filename = filename.lower()
  imgs = []
  if filename.endswith('.ofd'):
    ofd = OFD()
    if isinstance(file, str):
      with open(file, "rb") as f:
        ofdb64 = str(base64.b64encode(f.read()), "utf-8")
    else:
      ofdb64 = str(base64.b64encode(file), "utf-8")
    # print(ofdb64)
    ofd.read(ofdb64, save_xml=False,
             xml_name=f"{os.path.split(filename)[0]}_xml")
    img_np = ofd.to_jpg()
    ofd.del_data()
    for index in range(img_np.__len__()):
      img = np.array(img_np[index])
      imgs.append(img)
    # img_pil = Image.fromarray(img)
    # (w, h) = img_pil.size
    # scale_factor = FIXED_WIDTH / w
    # resized_img_pil = img_pil.resize((FIXED_WIDTH, int(h * scale_factor)))
    # img = np.array(resized_img_pil)
  elif filename.endswith('.pdf'):
    if isinstance(file, str):
      doc = fitz.open(file)
    else:
      doc = fitz.open("pdf", file)
    if len(doc) == 0:
      return {}
    for page_num in range(len(doc)):
      page = doc.load_page(page_num)
      original_width = page.rect.width
      for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        (w, h) = img[2], img[3]
        if w / original_width > 0.5:
          continue
        doc._deleteObject(xref)
      scale_factor = FIXED_WIDTH / original_width
      pix = page.get_pixmap(matrix=fitz.Matrix(scale_factor, scale_factor))
      img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w,pix.n)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      imgs.append(img)
    # processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
  else:
    if isinstance(file, str):
      # img = cv2.imread(file)[:, :, ::-1]
      img = np.array(Image.open(file))
    else:
      img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.COLOR_BGR2RGB)
    imgs.append(img)
    # processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image(img)
  return imgs


@app.route('/invoice_ocr', methods=['POST'])
def invoice_ocr():
  uploaded_file = request.files['file']
  print(uploaded_file.content_type)
  if uploaded_file is None or uploaded_file.content_type not in types:
    return {}
  read = uploaded_file.read()
  filename = uploaded_file.filename
  ocr_result = {}
  imgs = __get_img__(filename, read)
  for img in imgs:
    img2 = Image.fromarray(img)
    processed_img, orig_size, new_size, paste_coords, resize_img = preprocess_image2(img2, 640, 640)
    converted_detections, item_infos, item_boxes, items = predict2.start(resize_img)
    enhancer = ImageEnhance.Contrast(img2)
    img2 = enhancer.enhance(2.0).convert('L')
    img = np.array(img2)
    ocr_and_set_value(converted_detections, img, new_size, ocr_result, orig_size)
    item_result_list = []
    for item in item_infos:
      item_result = {}
      item_result_list.append(item_result)
      ocr_and_set_value(item, img, new_size, item_result, orig_size)
    if item_result_list.__len__() > 0:
      ocr_result['items'] = item_result_list

  return ocr_result

def process_single_object(obj, orig_size, new_size, img, det_true_list):
    """处理单个检测对象，返回裁剪后的图像、标签及是否启用检测的标志"""
    # 解析坐标和标签
    left, top, right, bottom = obj[0], obj[1], obj[2], obj[3]
    label = str(obj[4])

    # 坐标转换
    box = convert_coordinates([left, top, right, bottom], orig_size, new_size)
    left = max(0, math.floor(box[0]))
    top = max(0, math.floor(box[1] - 1))
    right = min(img.shape[1], math.ceil(box[2] + 1))
    bottom = min(img.shape[0], math.ceil(box[3] + 1))

    # 检查有效区域
    if (right <= left) or (bottom <= top):
      return None, None, None

    # 裁剪图像
    cropped_img = img[top:bottom, left:right]
    if cropped_img.size <= 0:
      return None, None, None

    # 判断是否需要检测增强
    is_det = label in det_true_list

    # 添加白色边框（仅针对需要检测的图片）
    if is_det:
      cropped_img = cv2.copyMakeBorder(
        cropped_img,
        top=5, bottom=5, left=5, right=5,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
      )

    # 统一通道格式
    if len(cropped_img.shape) == 2:
      cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
    elif cropped_img.shape[2] == 4:
      cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2RGB)
    else:
      cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    return cropped_img, label, is_det

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
# 主处理函数
def process_with_thread_pool(converted_detections, orig_size, new_size, img, det_true_list):
  """使用线程池并行处理所有检测对象"""
  det_group = {"det_true_img": [], "det_true_label": [], "det_false_img": [], "det_false_label": []}
  # for obj in converted_detections:
  #   cropped_img, label, is_det = process_single_object(obj, orig_size, new_size, img, det_true_list)
  #   if cropped_img is not None:
  #     group_key = "det_true" if is_det else "det_false"
  #     det_group[group_key + "_img"].append(cropped_img)
  #     det_group[group_key + "_label"].append(label)
  # 提交所有任务
  futures = [
    executor.submit(
      process_single_object,
      obj, orig_size, new_size, img, det_true_list
    )
    for obj in converted_detections
  ]

  # 收集结果
  for future in futures:
    cropped_img, label, is_det = future.result()
    if cropped_img is not None:
      group_key = "det_true" if is_det else "det_false"
      det_group[group_key + "_img"].append(cropped_img)
      det_group[group_key + "_label"].append(label)

  return det_group

def ocr_and_set_value(converted_detections, img, new_size, ocr_result, orig_size):
  start_time  = time.perf_counter()
  result_groups  = process_with_thread_pool(converted_detections, orig_size, new_size, img, det_true_list)
  det_img = result_groups.get("det_false_img", [])
  det_label = result_groups.get("det_false_label", [])

  if det_img:
    rr = ocr.ocr(det_img, det=False, cls=False)
    if rr:
      for line in rr:
        if line:
         for index, word_info in enumerate(line):
          if word_info and len(word_info) > 1:
            ocr_result.setdefault(det_label[index], '')
            ocr_result[det_label[index]] += word_info[0]

  # ocr_and_set_det_true_value(ocr_result, det_img, det_label)
  det_img = result_groups["det_true_img"]
  det_label = result_groups["det_true_label"]
  ocr_and_set_det_true_value(ocr_result, det_img, det_label)
  end_time  = time.perf_counter()
  execution_time = end_time - start_time
  print(f"识别耗时: {execution_time} 秒")

def ocr_and_set_det_true_value(ocr_result, det_img, det_label, is_det=True):

  if det_img:
    for index, e in enumerate(det_img):
      rr = ocr.ocr(e, det=True, cls=False)
      set_result_value(is_det, det_label[index], ocr_result, rr)


def set_result_value(is_det, label, ocr_result, rr):
  texts = get_result_value(is_det, rr)
  if texts:
    ocr_result.setdefault(label, '')
    ocr_result[label] += ''.join(texts)


def get_result_value(is_det, rr):
  if rr:
    texts = [
      word_info[1][0] if is_det else word_info[0]
      for line in rr if line
      for word_info in line
      if word_info and len(word_info) > 1
    ]
  else:texts = None
  return texts


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
