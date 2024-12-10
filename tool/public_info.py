from pathlib import Path

def get_file_path(filepath):
  directory_path = Path(filepath)
  directory_path.mkdir(parents=True, exist_ok=True)
  return filepath
# 待标注的目标路径
directory = get_file_path("D:\发票\ofd")
# directory = get_file_path("D:\发票\发票3")
# directory = get_file_path(r"D:\发票\fapiao")
base_path = get_file_path("out")
# 标注信息保存路径.
target_path = get_file_path(base_path + '/img2')
# 转换的label信息目录
convert_path = get_file_path(base_path + "/convert_label/")
# 数据集划分目录
dataset_path = get_file_path(base_path+ "/mydataset")
# 缩放与原图可视化图片保存路径
contrast_path = get_file_path(base_path + '/flag')


label_info = {
        "title": 0,
        "invoice_code": 1,
        "invoice_number": 2,
        "issue_date": 3,
        "buyer_name": 4,
        "buyer_code": 5,
        "tax_exclusive_total_amount": 6,
        "tax_total_amount": 7,
        "tax_inclusive_total_amount": 8,
        "seller_name": 9,
        "seller_code": 10
      }
label_coco_info = [{label_info[label] + 1: label} for label in label_info]