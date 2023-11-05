# Heart attack risk predict  

## Problem Description  

### The data is taken from [Heart Attack Risk Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset/data)  

The heart attack risk prediction model developed from this large dataset is a major advance in cardiovascular health. Using a combination of demographic, lifestyle, and health characteristics, this predictive model predicts the likelihood of a heart attack, providing researchers and healthcare providers with a tool for a proactive strategy.  

Using predictive analytics and machine learning techniques, this model accounts for the complex interdependencies between multiple factors including age, cholesterol levels, blood pressure, smoking habits, physical activity, dietary preferences, and more. This broad feature set provides a robust framework for understanding the complex dynamics contributing to heart attacks.  

The use of this model not only assists in risk stratification, but also lays the foundation for health promotion, optimisation of health care resources and a proactive approach to cardiovascular health. This demonstrates a collaborative effort to expand our understanding of cardiovascular well-being and is a step towards a healthy future by reducing the burden of cardiovascular disease through informed decision-making and proactive health management.

### EDA and Model Description

The results of EDA and model fitting can be observed in the file **nothebook.ipynb**

The following conclusions can be drawn from the estimation results of multiple models and ensemble of models (VotingClassifier):  

1. **Gradient Boosting Classifier**:  
    - Best Parameters: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 20, 'learning_rate': 0.2}  
    - Accuracy on the test kit: 59.76%  
    - ROC AUC on the test kit: 0.51  
    
2. **Random Forest Classifier**:   
    - Best Parameters: {'n_estimators': 500, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None}  
    - Accuracy on the test kit: 61.85%  
    - ROC AUC on the test kit: 0.55  

3. **Voting Classifier**:   
    - Accuracy on the test kit: 63.31%  
    - ROC AUC on the test kit: 0.54  

Based on the presented results, **Gradient Boosting Classifier** and **Random Forest Classifier** demonstrated the best performance among all the models considered, while **Voting Classifier (ensemble)** showed comparable performance but slightly worse.

Taking the above into account, the **Gradient Boosting Classifier** model demonstrated the best accuracy among all considered models on the test dataset.

The choice of this model is justified by its better performance compared to other models on the test and validation datasets, although the results still need improvement, especially for class 1 prediction.
    
#### Feature importance analysis 

After training the model, the attributes that most affect the outcome were selected. You can observe the description below

#### Data set glossary (by column)

**Age** - Patient's age  
**Cholesterol** - The patient's cholesterol levelа  
**Heart Rate** - The patient's heart rate  
**Exercise Hours Per Week** - Number of hours of physical exercise per week   
**Sedentary Hours Per Day** - Hours of sedentary work per day  
**Income** - The patient's income level  
**BMI** - Body Mass Index (BMI) of the patient  
**Triglycerides** - patient's triglyceride level   
**Heart Attack Risk** - Presence of risk of heart attack (1: Yes, 0: No)  
**Sleep Hours Per Day** - Number of hours of sleep per day  
**Systolic** - This is the first or top number in blood pressure measurement  
**Diastolic** - This is the second or bottom number in a blood pressure reading  

### Installing and running the project

#### Running on a local machine
1. Clone the repository:
```
git clone https://github.com/Sharpylo/ml_zoomcamp.git
```
2. Navigate to the project directory:
```
cd Midterm_Project
```
3. Activate the virtual environment:
```
env\Scripts\activate
```
4. Install necessary dependencies:
```
pip install -r requirements.txt
```

### Usage
#### Running the Project
- To run the project, execute the following commands:
```
python predict.py
```
Expected Result:  
![Run webapp result](./images/run_webapp_result.png)
- For running the test suite:
```
python predict_test.py
```
Expected Result:  
![Predict result](./images/predict_result.png)


### Building and Running Docker Container
- Build the Docker image:
Run the following command in the terminal where the Dockerfile is located:
```
docker build -t midterm_project .
```
Expected Result:  
![Build dockerfile result](./images/build_dockerfile_result.png)
- Run the Docker container:
Run the container and map the port:
```
docker run -p 9696:9696 --name my_midterm_project midterm_project
```
Expected Result:  
![Run dockerfile result](./images/run_dockerfile_result.png)
- For running the test suite:
```
python predict_test.py
```
Expected Result:  
![Predict result](./images/predict_result.png)

### Observation: If you want to train a model

Make your changes in train.py and run the file using the command below.
```
python train.py
```
Expected Result:  
![Train result](./images/train_result.png)

### Demonstration of work:

https://github.com/Sharpylo/ml_zoomcamp/raw/main/Midterm_Project/images/work_demonstration.mp4