# Face-Mask-Recognition

## Overview
Welcome to the Face Recognition Project! This project enables face recognition using MTCNN for face detection, one-shot learning for recognition, and combining both for a holistic solution.. Follow the steps below to get started.

## Getting Started

### Step 1: Collect Images

- Collect images from people, each with at least 10 pictures, including both mask and non-mask variations.
- Place the images in the `data` folder and create a folder for each person's images. Add all images to their respective folders.

### Step 2: Crop Faces

Run the following script to crop the faces in the images, overwriting the old ones:

```bash
python crop_img.py
```

### Step 3: Extract Features
Run the following script to extract features from the cropped faces and save them to features.json:

```bash
python calc_features.py
```

### Step 4: Test the Model
Run the following script to deploy the model and test its performance:

```bash
python deploy.py
```
