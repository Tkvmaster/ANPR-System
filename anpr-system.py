import argparse
import time
import re
from random import randint
from typing import List, Optional, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from paddleocr import PaddleOCR
import norfair
from norfair import Detection, Paths, Tracker, Video

DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

OCR_TH = 0.2
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)



def get_best_ocr(ocr_res, score, track_id, df):
    final_ocr = ''
    if track_id in df.index:
        if len(ocr_res) < 6:
            if len(df.loc[track_id]['Number_Plate']) < 6:
                if df.loc[track_id]['conf'] < score:
                    df.at[track_id, 'Number_Plate'] = ocr_res
                    df.at[track_id, 'conf'] = score
                    return ocr_res
                return df.loc[track_id]['Number_Plate']
            else:
                if len(df.loc[track_id]['Number_Plate']) < 6:
                    df.at[track_id, 'Number_Plate'] = ocr_res
                    df.at[track_id, 'conf'] = score
                    return ocr_res

        else:
            if df.loc[track_id]['conf'] < score:
                df.at[track_id, 'Number_Plate'] = ocr_res
                df.at[track_id, 'conf'] = score
                return ocr_res
            else:
                return df.loc[track_id]['Number_Plate']

    else:
        df.loc[track_id] = [ocr_res, score]
        return ocr_res


def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

def clean(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def clean(img):
    """Preprocess image before OCR"""
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img


def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y- dim[1] - baseline), (x + dim[0], y), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y -baseline), FONT_FACE, FONT_SCALE, (0,255,0), THICKNESS, cv2.LINE_AA)


def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate, scores = [], []
    for result in ocr_result[0]:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1][0])
            scores.append(result[1][1])

    plate = ''.join(plate)
    plate = re.sub(r'\W+', '', plate)
    if not scores:
        plate = ''
        scores.append(0)
    return plate.upper(), max(scores)


def recognize_plate_easyocr(img, coords,reader,region_threshold):
    """recognize license plate numbers using paddle OCR"""
    # separate coordinates from box
    xmin, ymin = coords[0]
    xmax, ymax = coords[1]
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    try:
        nplate = clean(nplate)
    except:
        return '',0

    ocr_result = reader.ocr(nplate)

    text, score = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    return text, score


class YOLO:
    def __init__(self, weights, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load model
        self.model = torch.hub.load('./yolov5', 'custom', source ='local', path=weights,force_reload=True)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections

def yolo_detections_to_norfair_detections(yolo_detections: torch.tensor):
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []
    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        scores = np.array(
            [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
        )
        norfair_detections.append(
            Detection(
                points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
            )
        )

    return norfair_detections

def running_anpr(in_video, out_video, df, model, tracker, ocr):
    # Declaring variables for video processing.
    df.drop(df.index, inplace=True)
    cap = cv2.VideoCapture(in_video)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(out_video, codec, fps, (width, height))
    ct = 0
    # Initializing some helper variables.

    # Reading video frame by frame.
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            h, w = img.shape[:2]
            print(f"[INFO] Frame Count : {ct+1}")

            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            yolo_detections = model(img,conf_threshold=0.55)

            detections = yolo_detections_to_norfair_detections(yolo_detections)
            tracked_objects = tracker.update(detections=detections)
            # norfair.draw_boxes(img, detections)
            # norfair.draw_tracked_boxes(img, tracked_objects)

            for obj in tracked_objects:
              points = obj.estimate.astype(int)
              points = tuple(points)

              cv2.rectangle(img,points[0],points[1],(0,255,0),2)
              ocr_res, score = recognize_plate_easyocr(img, points, ocr, 0.2)
              text = get_best_ocr(ocr_res, score, obj.id, df)

              draw_label(img, text, points[0][0], points[0][1])


            out.write(img)
            # Increasing frame count.
            ct = ct + 1
        else:
            break
    out.release
    print('[INFO] Video Stream Ended')

def main(weights_path, in_video, out_video, csv_path):
	ocr = PaddleOCR(lang='en')
	df = pd.DataFrame(columns = ['track_id', 'Number_Plate', 'conf'])
	df = df.set_index('track_id')

	model = YOLO(weights_path)

	distance_function = "iou_opt"
	distance_threshold = (
	    DISTANCE_THRESHOLD_BBOX
	)

	tracker = Tracker(
	    distance_function=distance_function,
	    distance_threshold=distance_threshold,
	)

	running_anpr(in_video, out_video, df, model, tracker, ocr)

	df = df[df['conf'] > 0.2]
	df.to_csv(csv_path, index=False)




parser = argparse.ArgumentParser(description = 'Run ANPR system on video feed')
parser.add_argument('--weight', type=str, help='Weights of the yolov5 model')
parser.add_argument('--input', type=str, help='Path to the input video')
parser.add_argument('--output', type=str, default='tracker_output.mp4', help='Path to the output video')
parser.add_argument('--csv', type=str, default='number_plates.csv', help='Path to the output CSV file')
args = parser.parse_args()

if __name__ == '__main__':
	main(args.weight, args.input, args.output, args.csv)





