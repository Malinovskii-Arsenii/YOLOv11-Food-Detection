# YOLOv11 Food Detection

##  Task Overview

This project is a part of the test assignment for the **Python Developer (Computer Vision, ML)** position. The goal was to develop a food recognition model based on video data using **YOLOv11**, including full pipeline from preprocessing to evaluation.

---

##  Project Structure

```
.
├── results/                     # CSVs and plots of training results
├── runs/                        # YOLO training runs with weights and metrics
├── README.md                    # This file
├── extract_frames.ipynb         # Extract frames from raw videos and training notebook for YOLOv11
└── requirements.txt             # Python dependencies
```

---

##  1. Data Preparation

### 1.1 Video Frame Extraction

* Videos in `.MOV` format were processed.
* Extracted **638** frames at \~1 FPS using `cv2.VideoCapture`.

### 1.2 Annotation

* Annotated manually using [Roboflow Annotate](https://roboflow.com).
* Classes:

  * `tea`, `shashlik`, `greek_salad`, `lamb`, `borscht`, `caesar_salad`, `cheese_soup`

### 1.3 Augmentation

* Applied in Roboflow:

  * Flip: Horizontal, Vertical
  * Rotation: Between -15° and +15°
  * Saturation: Between -25% and +25%
  * Brightness: Between -20% and +20%
  * Noise: Up to 1.05% of pixels
   
### 1.4 Splitting

* Dataset was split as:

  * `train`: 88%
  * `val`: 8%
  * `test`: 4%

---

##  2. Model Training

### 2.1 First Experiment

```python
model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs',
    name='yolov11-dish',
    save_period=10
)
```

**Results:**

* Precision: 0.9910
* Recall: 0.9821
* mAP@0.5: 0.9902
* mAP@0.5:0.95: 0.8858
* Val box loss: 0.4415

### 2.2 Second Experiment

```python
model.train(
    epochs=30,
    imgsz=640,
    batch=16,
    lr0=0.002,
    optimizer='SGD',
    project='runs',
    name='yolov11-dish_v2',
    save_period=5
)
```

**Results:**

* Precision: 0.9941
* Recall: 0.9949
* mAP@0.5: 0.9940
* mAP@0.5:0.95: 0.9296
* Val box loss: 0.3329

---

##  3. Metrics & Visualization

All training metrics (loss, mAP, precision, recall) were saved and plotted:

* `results/results_v1.csv`
* `results/results_v2.csv`

Visual comparison shows consistent improvement on all metrics after hyperparameter tuning.

---

##  4. Insights & Challenges

* **Annotation** was the most time-consuming part.
* **Roboflow** greatly helped with fast annotation and augmentation.
* **Hyperparameter tuning** significantly improved model quality.

---

##  Execution Time

* Total hours: **\~20-22 hours**

  * Data preparation: 6h
  * Training and tuning: 8h
  * Results analysis & report: 6-8h

---

##  YOLO Experience

* Before this task: use of YOLOv8 for academic CV project and for commercial projects, full pipeline experience with YOLOv11, training, tuning, analysis.

---

##  How to Reproduce

1. Clone repo:

```bash
git clone https://github.com/Malinovskii-Arsenii/yolov11-food-detection
cd yolov11-food-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train model:

```python
from ultralytics import YOLO
model = YOLO('yolov11n.pt')
model.train(data='dataset/data.yaml', epochs=50, imgsz=640)
```

4. Run inference:

```python
results = model('path/to/image.jpg', save=True)
```
