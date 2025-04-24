import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
os.chdir(project_root)

import tool.auto_label as auto_label
import tool.divide_dataset as divide_dataset
import tool.to_yolo as to_yolo
import tool.yolo_to_coco as yolo_to_coco
import tool.public_info as public_info


if __name__ == "__main__":
  # 自动标注
  # auto_label.label(public_info)
  # 转yolo格式
  # to_yolo.to(public_info)
  # 划分数据集
  # divide_dataset.divide(public_info)
  # yolo转coco
  yolo_to_coco.to(public_info)