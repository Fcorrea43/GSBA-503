# GSBA-503
Object Detection Comparison: Faster R-CNN vs RetinaNet

This project compares two popular deep learning object detection models — Faster R-CNN and RetinaNet — using PyTorch’s pretrained COCO weights. The goal is to evaluate model performance across a collection of real-world images by measuring number of detected objects, confidence scores, brightness, and inference time.

This work was completed as part of a take-home image analytics assignment. 

Take Home assigment

Project Overview

The code loads two pretrained models:

Faster R-CNN (ResNet-50 + FPN) — a two-stage detector prioritizing accuracy

RetinaNet (ResNet-50 + FPN) — a one-stage detector prioritizing speed and confidence

Each image undergoes the same preprocessing and is evaluated by both models.
For every image/model pair we collect:

Number of detected objects

Average confidence score

Average brightness

Inference time (seconds)

All results are saved into a Pandas dataframe for later analysis and visualization.

📂 Project Structure
├── images/                # Folder containing input images
├── results.csv            # Model outputs (detections & metrics)
├── charts/                # Generated plots (optional)
├── Take Home assignment.html  # Full report (exported notebook)
└── detect.py              # Main evaluation script

How It Works
1. Load Models
model_frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model_retinanet = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")


Both models run in eval mode and use GPU if available.

2. Preprocess Images
transform = transforms.ToTensor()


Images are converted to PyTorch tensors.

3. Run Inference

For each image:

Transform

Move to device

Run detection

Filter predictions by confidence

Measure time

Compute brightness

4. Store Results

Results are appended into a Pandas dataframe and exported to CSV or visualized.

Key Findings

Faster R-CNN detects more objects across nearly all images, making it more thorough.

RetinaNet runs faster and often produces higher average confidence, but detects fewer objects.

Models perform worse on very bright or very dark images, suggesting lighting strongly impacts detection.

Example Output Table
image	model	num_objects	avg_score	brightness	time_sec
img 7.jpeg	Faster R-CNN	4	0.75	109.39	0.816
img 7.jpeg	RetinaNet	1	0.83	109.39	0.632
...	...	...	...	...	...
📈 Visualizations

This project can generate useful charts:

Average objects detected per model

Average confidence per model

Inference time comparison

If you'd like, I can add the code to automatically save these charts to the repo.

▶️ How to Run

Clone the repository

Install PyTorch + Torchvision

Add your images to images/

Run:

python detect.py


View results in results.csv

Dependencies

Python 3.10+

PyTorch

Torchvision

NumPy

Pandas

Pillow

License

MIT License (or specify your preferred license)

Author

Fernando Correa
MSBA — University of San Diego
