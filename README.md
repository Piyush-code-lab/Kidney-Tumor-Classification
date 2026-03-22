# Kidney Tumor Classification

A deep learning project to classify kidney CT scans into 4 categories:
- Cyst
- Normal  
- Stone
- Tumor

## Model
- Architecture: VGG16 with custom head (Transfer Learning)
- Accuracy: 83.15%
- Training: Google Colab T4 GPU

## How to run

### 1. Clone the repo
- git clone https://github.com/Piyush-code-lab/Kidney-Tumor-Classification.git
- cd Kidney-Tumor-Classification

### 2. Create environment
- conda create -n kidney_env python=3.11 -y
- conda activate kidney_env

### 3. Install dependencies
- pip install -e .
- pip install -r requirements.txt

### 4. Train the model
python main.py

### 5. Run the web app
python app.py

Open http://localhost:5000 in your browser