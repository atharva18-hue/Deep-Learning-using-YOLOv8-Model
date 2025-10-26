#  Animal Species Detection using YOLOv8 Model

##  Overview
This project detects **animal species in real time** using **YOLOv8 (You Only Look Once)** — a cutting-edge deep learning model for object detection.  
It takes live camera input or video files and instantly classifies animals such as Buffalo, Elephant, Rhino, Zebra, Cheetah, Fox, Jaguar, Tiger, Lion, Panda, and more.

---

##  Key Features
-  Real-time animal detection using webcam or video
-  YOLOv8 trained custom model integration
-  Streamlit web app interface for user-friendly interaction
-  Visualized detection results with bounding boxes and confidence scores
-  Organized folder structure for training, testing, and deployment

---

##  Technologies Used
| Category        | Tools / Libraries                     |
|-----------------|--------------------------------------|
| **Deep Learning** | YOLOv8, PyTorch                     |
| **Web Framework** | Streamlit                            |
| **Data Processing** | OpenCV, NumPy, Pandas              |
| **Visualization** | Matplotlib, Seaborn                  |
| **Environment** | Python 3.10+, Virtual Environment (venv) |

---



## Table of Contents
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Evaluation](#evaluation)
- [Web App](#web-app)
- [Contirbuting](#contributing)
- [Author](#author)

## Datasets
The dataset used in this project consists of labeled images of 10 different animal classes: Buffalo, Cheetahs, Deer, Elephant, Fox, Jaguars, Lion, Panda, Tiger, Zebra. You can find the datasets: 
- [Dataset 1](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)
- [Dataset 2](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)
- [Dataset 3](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset )

## Project Structure
    ├── config
    │   └── custom.yaml    
    ├── data
    │   ├── images         
    │   └── labels         
    ├── logs
    │   └── log.log      
    ├── notebooks
    │   └── yolov8.ipynb
    ├── runs
    │   └── detect
    │       ├── train
    │       └── val
    ├── scripts
    │   ├── app.py
    │   ├── convert_format.py
    │   └── train_test_split.py
    ├── README.md
    └── requirements.txt

## Getting Started
Follow theses steps to set up the environment and run the application..
2. Clone the forked repository.
    ```bash
    git clone https://github.com/<YOUR-USERNAME>/Animal-Species-Detection
    cd Animal-Species-Detection
    ```

3. Create a python virtual environment.
    ``` bash
    python3 -m venv venv
    ```

4. Activate the virtual environment.

    - On Linux and macOS
    ``` bash
    source venv/bin/activate
    ```
    - On Windows
    ``` bash
    venv\Scripts\activate
    ```

5. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
6. Run the application.
    ```python
    streamlit run './scripts/app.py'
    ```

## Evaluation
The performance of the model is evaluated by metrics such as Precision, Recal, and Mean Average Precision (mAP).

| Model   | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
|---------|-----------|--------|----------|---------|--------------|
| YOLOv8  |   0.944   |  0.915 |   0.93   |   0.95  |    0.804     |


## Web App
The trained model has been deployed on Hugging Face for practical use.
- you can access the deployed [web app](https://huggingface.co/spaces/ldebele/animal_detection_app)

## Contributing
Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or a pull request.

