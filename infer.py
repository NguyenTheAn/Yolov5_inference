import argparse
import os
import cv2
import torch
import numpy as np
from utils import check_img_size, letterbox, non_max_suppression, scale_coords

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
parser.add_argument('--img_path', type=str)
parser.add_argument('--img_size', type=int, default=416)
parser.add_argument('--show_image', action="store_true")
parser.add_argument('--save_image', action="store_true")

args = parser.parse_args()

path = args.img_path
device = torch.device("cuda", 0)
model = torch.load(args.weights, map_location=device)['model'].to(device).float()
stride = int(model.stride.max()) 
imgsz = check_img_size(args.img_size, s = stride)
origin = cv2.imread(path)
img= origin.copy()
img = letterbox(img, imgsz, stride)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).to(device).float()
img /= 255.0  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]

pred = model(img.to(device), augment=False, visualize=False)[0]
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
for i, det in enumerate(pred): 
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], origin.shape).round()

if args.show_image:
    for bbox in pred[0]:
        bbox = bbox.cpu().numpy()
        origin = cv2.rectangle(origin, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
    cv2.imshow("image", origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if args.save_image:
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/" + os.path.basename(path), origin)