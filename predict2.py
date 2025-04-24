import numpy as np
from deploy.python.infer import Detector
from rtree import index
from shapely.geometry import Point, box as Box
import time

# CPU OR GPU
device = 'CPU'

# 设置模型目录和输出目录
model_dir = r"models/rtdetrv2"  # 替换为你的模型目录
output_dir = r"output"  # 替换为你的输出目录
confidence_threshold = 0.3
model2 = Detector(model_dir=model_dir,
                 device=device,
                 run_mode='paddle',
                 batch_size=1,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir=output_dir,
                 threshold=confidence_threshold,
                 delete_shuffle_pass=False
                 )
labels = model2.pred_config.labels



def start(processed_img):
  start_time = time.perf_counter()
  img = np.array(processed_img).astype(np.uint8)
  results = model2.predict_image([img], visual=False)
  boxes = {}
  item_boxes = []
  items = []
  for e in results['boxes']:
    class_id, confidence, left, top, right, bottom = e
    if confidence < confidence_threshold:
      continue
    label = labels[int(class_id)]
    # n = (left, top, right, bottom, label, confidence)
    if label == 'item':
      # set_box(item_boxes, n)
     continue
    elif label.startswith('item_'):
      # set_box(items, n)
      continue
    else:
      boxe = boxes.get(label, None)
      if boxe is None or boxe[5] < confidence:
        boxe = [left, top, right, bottom, label, confidence]
        boxes[label] = boxe
        continue
  spatial_index, boxs = create_spatial_index(item_boxes)
  item_info = {}
  for item in items:
    point = ((item[0] + item[2]) / 2, (item[1] + item[3]) / 2)
    i = point_in_boxes(point, boxs, spatial_index)
    if i is not None:
      if i in item_info:
        item_info[i].append(item)
      else:
        item_info[i] = [item]
  end_time  = time.perf_counter()
  execution_time = end_time - start_time
  print(f"检测耗时: {execution_time} 秒")
  return boxes.values(), item_info.values(), item_boxes, items

def set_box(item_boxes, n):
  new_box = None
  for box in item_boxes:
    new_box = _get_box(n, box)
    if new_box is not None:
      break
  if new_box is not None:
    if new_box[5] >= n[5]:
      return
    new_box[0] = n[0]
    new_box[1] = n[1]
    new_box[2] = n[2]
    new_box[3] = n[3]
    new_box[4] = n[4]
    new_box[5] = n[5]
    # new_box = [left, top, right, bottom, label, confidence]
  else:
    item_boxes.append(n)
def _get_box(box1, box2):
  if not (box2[2] < box1[0] or box2[0] > box1[2] or box2[3] < box1[1] or box2[1] > box1[3]) :
    return box2
  else: return None

def create_spatial_index(boxes):
    """
    创建一个空间索引来存储边界框。

    参数:
    boxes -- 边界框列表，每个边界框是一个四元素元组 (left, top, right, bottom)

    返回:
    RTree 索引对象
    """
    boxs = []
    idx = index.Index()
    for i, (left, top, right, bottom, label, confidence) in enumerate(boxes):
      # 插入边界框到索引中
      boxs.append((left, top, right, bottom))
      idx.insert(i, (left, top, right, bottom))
    return idx, boxs


def point_in_boxes(point, boxes, spatial_index):
  """
  使用空间索引快速查找包含给定坐标的边界框。

  参数:
  point -- 一个元组，表示要检查的点 (x, y)
  boxes -- 边界框列表
  spatial_index -- RTree 索引对象

  返回:
  如果点位于某个边界框内，则返回该边界框；否则返回None。
  """
  p = Point(point)
  for i in spatial_index.intersection((point[0], point[1], point[0], point[1])):
    b = Box(*boxes[i])
    if p.within(b):
      return i
  return None