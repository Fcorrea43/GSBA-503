# GSBA-503
# Object Detection Comparison: Faster R-CNN vs RetinaNet

This project compares two widely used deep learning object detection models, Faster R-CNN and RetinaNet, using PyTorch’s pretrained COCO weights. The purpose is to evaluate model performance across a set of real-world images by measuring the number of detected objects, confidence scores, brightness, and inference time.

This work was completed as part of a take-home image analytics assignment.

---

## Project Overview

The code loads and evaluates two pretrained object detection models:

- Faster R-CNN (ResNet-50 + FPN): a two-stage detector focused on accuracy
- RetinaNet (ResNet-50 + FPN): a one-stage detector optimized for speed and confidence

For each input image, both models produce predictions. The script records the following metrics:

- Number of detected objects
- Average detection confidence
- Average image brightness
- Inference time (seconds)

Outputs are stored in a Pandas dataframe for further analysis.

---

## Project Structure

```
├── images/                         # Folder containing the input images
├── detect.py                       # Main script for running detection
├── results.csv                     # Exported results table
├── Take Home assignment.html       # Exported HTML notebook (assignment write-up)
└── charts/                         # Optional folder for generated plots
```

---

## How the Code Works

### 1. Load Pretrained Models
```python
model_frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model_retinanet = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
```

### 2. Move Models to CPU or GPU
The script automatically detects whether CUDA is available.

### 3. Image Preprocessing
Each image is converted into a PyTorch tensor using `transforms.ToTensor()`.

### 4. Inference Loop
For every image in the `images/` directory, the script:

- Loads and transforms the image
- Performs inference with both models
- Extracts bounding boxes, confidence scores, and object counts
- Measures inference time
- Computes average brightness

### 5. Save Results
All metrics are saved into a dataframe and exported as `results.csv`.

---

## Key Findings

- Faster R-CNN consistently detected more objects per image, showing stronger overall accuracy.
- RetinaNet produced fewer detections but achieved slightly higher average confidence and faster runtime.
- Both models performed worse on very dark or very bright images, indicating that lighting conditions influence detection quality.

---

## Example Results Table

| image        | model        | num_objects | avg_score | brightness | time_sec |
|--------------|--------------|-------------|-----------|------------|----------|
| img7.jpeg    | Faster R-CNN | 4           | 0.75      | 109.39     | 0.816    |
| img7.jpeg    | RetinaNet    | 1           | 0.82      | 109.39     | 0.632    |
| ...          | ...          | ...         | ...       | ...        | ...      |

---

## How to Run the Script

1. Clone the repository.
2. Install dependencies, including PyTorch and Torchvision.
3. Place your image files inside the `images/` folder.
4. Run:

```
python detect.py
```

5. Review the generated `results.csv` file.

---

## Dependencies

- Python 3.10 or higher
- PyTorch
- Torchvision
- NumPy
- Pandas
- Pillow

---

## Author

Fernando Correa  
MSBA, University of San Diego

