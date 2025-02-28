# Jurassic Park Character Face Recognition

## Overview
This project is an **AI-powered face detection and recognition system** designed to identify characters from the Jurassic Park franchise. It utilizes **deep learning techniques**, including **RetinaFace** for face detection and a **custom CNN** for character classification, to accurately detect and label characters from movie scenes and promotional images.

## Dataset
### **Source**
- **Custom scraped dataset** from Jurassic Park movie scenes and promotional images.
- **Publicly available character image datasets** (if applicable).
- **Manually annotated labels** for supervised learning.

### **Target Output**
- **Face bounding boxes** with confidence scores.
- **Recognized characters** (e.g., Dr. Alan Grant, Dr. Ellie Sattler, Dr. Ian Malcolm, John Hammond, etc.).

## How to Use the Project
### **1. Installation**
#### **Prerequisites**
Ensure you have **Python 3.8+** and the required dependencies:
```bash
pip install -r requirements.txt
```

#### **Project Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/jurassic-park-recognition.git
   cd jurassic-park-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **2. Running Face Detection**
Run character detection on an image:
```bash
python main.py --image test.jpg
```

### **3. Training the Model**
To train the character recognition model from scratch:
```bash
python src/functions/train.py
```

### **4. Running Face Recognition on a Dataset**
To process an entire dataset of images and recognize characters:
```bash
python src/functions/detect.py --dataset_path src/dataset/
```

### **5. Deploying the Model**
To deploy the trained model for inference:
```bash
python src/functions/deploy.py --model_path models/face_recognition_model.h5
```

## Repository Structure
```
JURASSIC-PARK-RECOGNITION/
│── results/               # Stores detected face outputs
│   ├── detected_faces.csv
│   ├── preprocessed_images.npy
│── src/                   # Source code directory
│   ├── dataset/           # Training dataset
│   ├── functions/         # Core functionality
│   │   ├── deploy.py
│   │   ├── detect.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── utils.py
│── models/                # Pre-trained and trained models
│   ├── CNN.py
│   ├── character_recognition_model.h5
│   ├── label_encoder.pkl
│── main.py                # Main execution script
│── environments.yml        # Conda environment setup
│── README.md              # Project documentation
│── requirements.txt        # Python package dependencies
```
