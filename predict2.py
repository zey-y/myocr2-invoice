from ultralytics import YOLOv10
from torchvision import transforms

model = YOLOv10(model=r'models/best.pt')

def start(processed_img):
  transform = transforms.ToTensor()
  processed_img = transform(processed_img).unsqueeze(0)
  results = model(processed_img)[0]
  boxes = results.boxes.data.tolist()
  names = results.names

  converted_detections = []
  labels = {}
  for obj in boxes:
    left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(
        obj[3])
    label = int(obj[5])
    if label in labels:
      continue
    labels[label] = label
    converted_detections.append([left, top, right, bottom, names[label], obj[4]])
  return converted_detections
