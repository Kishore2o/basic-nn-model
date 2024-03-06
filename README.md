# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: S.Kishore
### Register Number: 212222240050
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('StudentsData').sheet1
data=worksheet.get_all_values()
print(data)

StudentsData = pd.DataFrame(data[1:], columns=data[0])
StudentsData = StudentsData.astype({'INPUT':'float'})
StudentsData = StudentsData.astype({'OUTPUT':'float'})

X = StudentsData[['INPUT']].values
y = StudentsData[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 30)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

ai_model=Sequential([
    Dense(units=8,activation='relu',input_shape=[1]),
    Dense(units=9,activation='relu'),
    Dense(units=1)
])

ai_model.compile(optimizer='rmsprop',loss='mse')

ai_model.fit(X_train1,y_train,epochs=20)

loss_df = pd.DataFrame(ai_model.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
ai_model.evaluate(X_test1,y_test)

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_model.predict(X_n1_1)



```
## Dataset Information

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/a811ee65-fa6f-46a7-b389-e7eda21f79ec)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/af8b8cad-0f5c-44fd-9994-40bea9235623)


### Test Data Root Mean Squared Error

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/477739e8-9481-4f1f-aeb9-71a56e750115)


### New Sample Data Prediction

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/c4eb4d5c-c6f4-4cb3-ae58-6799758623b6)


## RESULT


A neural network regression model for the given dataset has been developed Sucessfully.
