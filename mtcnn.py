from facenet_pytorch import MTCNN
import torch
from PIL import Image

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

def getFace(img_path):
    img = Image.open(img_path)
    boxes, _ = mtcnn.detect(img)
    if len(boxes) > 0:
        boxes = boxes[0].astype(int)
        x = boxes[0]
        y = boxes[1]
        w = boxes[2]
        h = boxes[3]
        img_crop = img.crop((x,y,w,h))
    return img_crop

