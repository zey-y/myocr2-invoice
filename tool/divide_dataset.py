import os
import random
import shutil


def copy_files(src_dir, dst_dir, filenames, extension):
  os.makedirs(dst_dir, exist_ok=True)
  missing_files = 0
  for filename in filenames:
    src_path = os.path.join(src_dir, filename + extension)
    dst_path = os.path.join(dst_dir, filename + extension)

    # Check if the file exists before copying
    if os.path.exists(src_path):
      shutil.copy(src_path, dst_path)
    else:
      print(f"Warning: File not found for {filename}")
      missing_files += 1

  return missing_files


def split_and_copy_dataset(image_dir, label_dir, output_dir, train_ratio=0.7,
    valid_ratio=0.15, test_ratio=0.15):
  # 获取所有图像文件的文件名（不包括文件扩展名）
  image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]

  # 随机打乱文件名列表
  random.shuffle(image_filenames)

  # 计算训练集、验证集和测试集的数量
  total_count = len(image_filenames)
  train_count = int(total_count * train_ratio)
  valid_count = int(total_count * valid_ratio)
  test_count = total_count - train_count - valid_count

  # 定义输出文件夹路径
  train_image_dir = os.path.join(output_dir, 'train', 'images')
  train_label_dir = os.path.join(output_dir, 'train', 'labels')
  valid_image_dir = os.path.join(output_dir, 'valid', 'images')
  valid_label_dir = os.path.join(output_dir, 'valid', 'labels')
  test_image_dir = os.path.join(output_dir, 'test', 'images')
  test_label_dir = os.path.join(output_dir, 'test', 'labels')

  # 复制图像和标签文件到对应的文件夹
  train_missing_files = copy_files(image_dir, train_image_dir,
                                   image_filenames[:train_count], '.png')
  train_missing_files += copy_files(label_dir, train_label_dir,
                                    image_filenames[:train_count], '.txt')

  valid_missing_files = copy_files(image_dir, valid_image_dir, image_filenames[
                                                               train_count:train_count + valid_count],
                                   '.png')
  valid_missing_files += copy_files(label_dir, valid_label_dir, image_filenames[
                                                                train_count:train_count + valid_count],
                                    '.txt')

  test_missing_files = copy_files(image_dir, test_image_dir,
                                  image_filenames[train_count + valid_count:],
                                  '.png')
  test_missing_files += copy_files(label_dir, test_label_dir,
                                   image_filenames[train_count + valid_count:],
                                   '.txt')

  # Print the count of each dataset
  print(
    f"Train dataset count: {train_count}, Missing files: {train_missing_files}")
  print(
    f"Validation dataset count: {valid_count}, Missing files: {valid_missing_files}")
  print(
    f"Test dataset count: {test_count}, Missing files: {test_missing_files}")

def divide(public_info):
  split_and_copy_dataset(public_info.target_path, public_info.convert_path, public_info.dataset_path)

