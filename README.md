# Detecting Drowsiness using Object Detection & CNN's

**Authors:** Abdulrasol Alofi, Sarim Mundres, and Isa Bukhari  
**Department:** Computer Science, University of California - Davis

## Abstract
Driving is an essential part of modern life, yet driver drowsiness significantly increases the risk of accidents. Our project aims to develop an advanced system capable of detecting driver drowsiness through real-time video stream analysis using computer vision and supervised machine learning techniques. The system will issue incrementing audible alerts to enhance road safety by preventing accidents caused by driver fatigue.

## Introduction

### Problem Description
Driving carries inherent risks, especially when drivers are drowsy. Drowsiness impairs reaction times, reduces attention, and increases the likelihood of accidents. The AAA Foundation for Traffic Safety reports that driver drowsiness is responsible for approximately 328,000 crashes each year, highlighting the need for effective drowsiness detection systems.

### Motivation
The motivation for this project is to enhance road safety by addressing driver drowsiness. By developing a system that detects drowsiness in real-time and alerts the driver, we aim to reduce the number of accidents caused by fatigue and save lives.

### Dataset
We utilize a publicly available dataset from Kaggle, containing labeled images of subjects in various states of alertness, including yawning and eye states. This dataset is crucial for training our supervised learning models to accurately detect drowsiness markers. The dataset can be accessed [here](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset).

### Proposed Method
Our proposed method involves:
- Preprocessing the dataset for training.
- Training supervised learning models to recognize drowsiness markers such as yawning, facial positioning, and eye movements.
- Implementing object detection algorithms to locate and classify these markers in real-time video streams.
- Issuing incrementing audible alerts when drowsiness is detected and recommending rest stops if repeated drowsiness is observed.

### Contribution
- **Abdulrasol Alofi:** Implemented CNN’s, Project Demo, Eye/Face Detection, and Report (Abstract, Introduction, Background & Methods)
- **Sarim Mundres:** Implemented Face Detection and Report (Methods & Discussion)
- **Isa Bukhari:** Implemented Initial Eye Detection and Report (Results & Conclusion)

### Paper Organisation
The paper consists of:
- **Background:** Overview of existing research and technologies related to driver drowsiness detection.
- **Data Exploration & Analysis:** Examination and preprocessing of the dataset.
- **Methods:** Description of the supervised learning and object detection techniques applied.
- **Experimental Results:** Performance and outcomes of our drowsiness detection models.
- **Discussion:** Interpretation of the results, implications, and effectiveness of our approach.
- **Conclusion:** Summary of key findings, contributions, and suggestions for future work.

## Background

### Key Concepts and Terms
- **Drowsiness:** A state of strong desire for sleep or falling asleep, impairing reaction times and attention.
- **Cascades:** Series of classifiers applied sequentially for object detection.
- **Object Detection:** Identifying and locating objects within an image or video stream.
- **Supervised Learning:** Training models on a labeled dataset to map inputs to outputs.
- **Neural Networks (NNs):** Computational models inspired by the human brain, capable of learning complex patterns.
- **Convolutional Neural Networks (CNNs):** Deep neural networks designed for processing images.
- **Real-Time Processing:** Immediate feedback or results from data processing.

## Methods

### Loading and Organizing Images
Images are loaded from specified directories and stored based on labels.

### Converting Images to Grayscale
Simplifies data by reducing color channels, retaining essential features for model training.

### Detecting and Extracting Face Regions
Focuses on the most relevant part for yawning detection using OpenCV’s `detectMultiScale` method.

### Resizing Face Images
Ensures consistent input dimensions for the neural network.

### Preparing Data for Model Training
Converts images to numpy arrays and shuffles the data, creating labels for supervised learning.

### Splitting and Normalizing Data
Splits data into training and testing sets with an 80-20 split, normalizing pixel values to a range of 0 to 1.

### Building and Training the CNN
The CNN model consists of several layers, each serving a specific purpose.

### Evaluating the Model
Evaluates metrics such as accuracy, f1 score, precision, and recall, guiding further refinement.

## Results

### Eye Model Results
Achieved high training accuracy (98.6%) and validation accuracy (94.5%), effectively distinguishing between open and closed eyes.

### Yawn Model Results
Achieved high training accuracy (99.6%) and validation accuracy (93.2%), effectively classifying yawning.

## Discussion

### Challenges
Faced challenges in differentiating between right and left eyes, achieving over 90 percent accuracy, and robust face detection in varied conditions.

### Future Improvements
Include using facial feature markers for more accurate results and implementing an add-on to display local rest spots via REST APIs.

## Conclusion
Developed an effective driver drowsiness detection system using computer vision and supervised machine learning techniques. Despite challenges, demonstrated the feasibility and reliability of our approach in both controlled and real-world environments. Future improvements aim to further elevate the utility and impact of our drowsiness detection system.

## Acknowledgements
This research received support during the ECS 174: Computer Vision course, instructed by Professor Hamed Pirsiavash, PhD & M.Sc. at the College of Engineering, University of California - Davis.

## Works Cited
- Tefft, B.C. (2024). Drowsy Driving in Fatal Crashes, United States, 2017–2021 (Research Brief). Washington, D.C.: AAA Foundation for Traffic Safety.


### Model Download Links
- [CNN_eye_normal.keras](https://store9.gofile.io/download/direct/49cfe719-aaba-407a-9bed-8841b44d2141/CNN_eye_normal.keras)
- [yawn_detection_model.keras](https://store9.gofile.io/download/direct/d1826209-b855-4b2d-aab2-430e850174c2/yawn_detection_model.keras)
