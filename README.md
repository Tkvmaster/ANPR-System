<h1 align="center">ANPR System</h1>

<div align= "center"><img src="utility_files/anpr_logo.jpg" width="200" height="200"/>
  <h4>Automatic number plate detection system using Deep Learning and Computer Vision concepts in order to detect license numbers of vehicles in real-time video streams.</h4>
</div>


![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tkvmaster/ANPR-System/issues)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/tkvmaster/)

## :innocent: Overview
Automatic number-plate recognition (`ANPR`) is a technology that uses optical character recognition on images to read vehicle registration plates to create vehicle location data. It may be used in a variety of public settings to serve a variety of functions, including automatic toll tax collection, car park systems, and automatic vehicle parking systems.

This project uses YOLOv5 for number plate detection and paddleocr for recognizing characters of the detected number plate. It also uses object tracking to track number plates and get the best OCR result for each plate, which is then saved into a CSV file.

## :wrench: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [YOLOv5](https://github.com/rkuo2000/yolov5)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [NorFair](https://tryolabs.github.io/norfair/2.1.1/)

## :page_with_curl: Proposed system
To solve this problem, I've taken a two-step approach. In the first step is number plate detection model is trained using YOLOv5 model.

The second step is to run the number plate detection model to locate all of the number plate present in an image and extract plate region from that image. Once a number plate is located, preprocessing is performed on ROI image and PaddleOCR is used to recognise characters in the number plate.

When running above proposed ANPR on a video, it causes some issues which makes the ANPR less accurate, such as Jittering, Fluctuation of OCR output. But if the tracker is used, these issues can be rectified.  The NorFai Tracker will be used to track the number plate and ensure the best OCR results are obtained. The recognized number plates will be saved to a CSV file.

## :file_folder: Dataset
The dataset consists of 928 images different types of vehicles.
These images are collected from:
 
- <a href="https://www.kaggle.com/datasets/andrewmvd/car-plate-detection">Car License Plate Detection Dataset</a>
- <a href="https://www.kaggle.com/datasets/andrewmvd/car-plate-detection">Automatic Number Plate Recognition Dataset</a>
- <a href="https://github.com/Tkvmaster/ANPR-System/blob/main/image_scrapping.ipynb">Web Scraping Images from Google</a>

## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/Tkvmaster/ANPR-System/blob/main/requirements.txt)

Make sure to install correct gpu versions of PaddlePaddle and torch.


## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/Tkvmaster/ANPR-System.git
```

2. Change your directory to the cloned repo 
```
$ cd ANPR-System
```

3. clone the YOLOv5 directory from GitHub
```
$ git clone https://github.com/rkuo2000/yolov5/
```


4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

To detect number plates in video feed: 
```
$ python3 anpr-system.py \
    --weights yolo_weights.pt \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --csv all_number_plates.csv
```


## :key: Results

![Alt Text](utility_files/tracker_output.gif)



Video Source : https://www.videvo.net/video/cars-driving-along-an-indian-freeway/6374/

## :warning: Limitations
This ANPR system is designed for real-time license plate detection and recognition, but it may not work well in all situations. The accuracy of the system may be affected by factors such as lighting, camera quality, and the angle and position of the license plate in the video.

## :eyes: Licensing
The code in this project is licensed under [MIT License](LICENSE).