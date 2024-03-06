# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

In this model we will discuss with a neural network with 3 layers of neurons excluding input . First hidden layer with 8 neurons , Second hidden layer with 9 neurons and final Output layer with 1 neuron to predict the regression case scenario.

## Neural Network Model

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/85e7e40a-29d9-4e2f-9976-25a3a00d8020)


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

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/4390fda2-db4c-4a14-bc5c-1410c9ea70b2)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/64691c24-57aa-4201-a4ed-a45e8f3a723f)


### Test Data Root Mean Squared Error

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/60f679bf-09ec-453c-ac21-d475c980dd53)


### New Sample Data Prediction

![image](https://github.com/Kishore2o/basic-nn-model/assets/118679883/622ca6fc-f8fd-4820-89f7-b8df111aee77)


## RESULT


A neural network regression model for the given dataset has been developed Sucessfully.
