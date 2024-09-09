# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step1. Import the required packages and print the present data.

Step2. Print the placement data and salary data.

Step3. Find the null and duplicate values.

Step4. Using logistic regression find the predicted values of accuracy , confusion matrices.

Step5. Display the results.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GOBATHI P
RegisterNumber: 212222080017

```py
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
![image](https://github.com/user-attachments/assets/51af5a84-bfb6-4d4b-ac67-1be6d9ad1dd0)

![image](https://github.com/user-attachments/assets/faa4025d-d39d-456e-8bc7-11d5463890c9)

![image](https://github.com/user-attachments/assets/504b292f-6d5c-493c-a67d-cbf416bf3001)

![image](https://github.com/user-attachments/assets/b39b0d4e-bd2e-4231-8000-bbce85ab2fd3)

![image](https://github.com/user-attachments/assets/0bd2e2dc-841d-4ea9-951d-15ee22f3c4bd)

![image](https://github.com/user-attachments/assets/8ab2b5d4-c258-481c-9ee9-9f95e0ad2b03)

![image](https://github.com/user-attachments/assets/c505b508-d766-4093-86cd-f1520ff6844b)

![image](https://github.com/user-attachments/assets/5beac3f6-99c9-40a3-8633-0651acb98d8f)

![image](https://github.com/user-attachments/assets/71fdcc23-8491-49be-886a-59cb2b9627c0)

![image](https://github.com/user-attachments/assets/c911c47a-e017-4e84-837c-5c75b82f12a8)

![image](https://github.com/user-attachments/assets/5d0d51ac-531d-4fdf-8146-85e5b7685e6c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
