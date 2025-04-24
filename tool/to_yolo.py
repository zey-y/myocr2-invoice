import os
import cv2
from PIL import Image

# filename = "D:/Project/python/img2/"  # 需要转换的标注文件

"""
# 读取数据文件
"""


def read_txt(file_name):
  data = []
  file = open(file_name, 'r', encoding='utf-8')
  file_data = file.readlines()  # 读取所有行
  for row in file_data:
    # tmp_list = row.split(' ')
    data.append(row)  # 将每行数据插入data中
  file.close()
  return data


"""
# 创建文件夹
"""


def create_folder(filename):
  filename = filename.strip()
  filename = filename.rstrip("\\")
  isExists = os.path.exists(filename)

  if not isExists:
    os.makedirs(filename)
    print(filename + "创建成功")
    return True
  else:
    print(filename + "已存在")
    return False


"""
# 创建txt标注文件
"""


def txt_create(public_info, name, msg):
  desktop_path = public_info.convert_path  # 新创建的txt文件的存放路径
  full_path = desktop_path + "/" + name.split('.')[0] + '.txt'

  if os.path.exists(full_path):
    file = open(full_path, 'a', encoding='utf-8')
    file.write(msg + '\n')
    file.close()
  else:
    file = open(full_path, 'w', encoding='utf-8')
    file.write(msg + '\n')
    file.close()


"""
# 坐标归一化
"""


def normalization(xmin, ymin, xmax, ymax, img_w, img_h):
  x = round((xmin + xmax) / (2.0 * img_w), 6)
  y = round((ymin + ymax) / (2.0 * img_h), 6)
  w = round((xmax - xmin) / (1.0 * img_w), 6)
  h = round((ymax - ymin) / (1.0 * img_h), 6)
  return x, y, w, h


"""
# 每张图片的尺寸
"""


def get_img_size(img_path):
  print("yolo_label_url:", img_path)
  mat = Image.open(img_path)
  # mat = cv2.imread(img_path)
  if mat is None:
    return mat
  # return mat.shape
  return mat.size

def to(public_info):
  # path = "D:\Project\python\myocr2-invoice\img2"
  # 原始数据表  #需要转换的标注文件
  data = read_txt(public_info.target_path + "/" + "Label.txt")
  for img_data in data:
    # 每行数据提取图片名
    img_name = img_data.split("\t")[0].split('/')[1]
    img_data = img_data.split(" ")
    # 每行数据提取坐标信息
    xywh = (str(img_data).split("\\t")[1].split("\\n")[0]
            .replace("',", "")
            .replace("'", "").strip('[')
            .strip(']')
            .replace("false", "1")
            .replace("true", "1"))
    # print(xywh)
    xywh = eval(xywh)  # 转换为元组或者字典
    img_size = get_img_size(public_info.target_path + "/" + img_name)
    if img_size is None:
      continue
    if type(xywh) == tuple:

      for xy_line in xywh:
        yolo_data = normalization(xy_line['points'][0][0],
                                  xy_line['points'][0][1],
                                  xy_line['points'][2][0],
                                  xy_line['points'][2][1], img_size[1],
                                  img_size[0])  # 处理每个字典的坐标信息，转换为归一化后的yolo标注格式
        # print("0 %s %s %s %s" % (str(yolo_data[0]) , str(yolo_data[1]) , str(yolo_data[2]) , str(yolo_data[3])))
        print(img_name)
        txt_create(public_info, img_name, "%s %s %s %s %s" % (str(public_info.label_info[str(xy_line['transcription'])]),
          str(yolo_data[0]), str(yolo_data[1]), str(yolo_data[2]),
          str(yolo_data[3])))
    if type(xywh) == dict:
      yolo_data = normalization(xywh['points'][0][0], xywh['points'][0][1],
                                xywh['points'][2][0],
                                xywh['points'][2][1], img_size[1], img_size[0])
      txt_create(public_info, img_name,
                 "0 %s %s %s %s" % (
                 str(yolo_data[0]), str(yolo_data[1]), str(yolo_data[2]),
                 str(yolo_data[3])))
      # print("0 %s %s %s %s" % (str(yolo_data[0]) , str(yolo_data[1]) , str(yolo_data[2]) , str(yolo_data[3])))
      print(img_name)
