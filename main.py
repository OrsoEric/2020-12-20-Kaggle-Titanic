import numpy as lib_numpy
import pandas as lib_pandas
import tensorflow as lib_tensorflow
from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.layers import Dense, Activation, Dropout

#-------------------------------------------------------------
# Settings
#-------------------------------------------------------------

#with estimate at 0, will skip this step and use a post process data.csv
gi_age_estimate_epoch = 500
gi_survived_epoch = 1000

print("---------------------------------------------------------")
print("Hello World")

#-------------------------------------------------------------
# Construct data set to be used for training, validation, and prediction
#-------------------------------------------------------------
#   input: data that serve as the input for the machine learning
#   output: data that serve as the output for the machine learning
#   train: data used to train the ML model
#   validation: data used to estimate accuracy of the model
#   prediction: data for which output is not provided and needs to estimated by the ML model

#Training data CSV column description
#   PassengerID | unique ID of passenger. 2/3 of the list has known survival status in train
#   Survived | true = survived | false = died | < this is what I want my neural network to compute from test data
#   Pclass | Socio economic factor | 1 = rich | 2 = normal | 3 = poor
#   Name | passenger name. Scrub from raw as name should not influence survival rate
#   Sex | male/female | sex is strongly correlated with survival
#   Age | years | there are unknowns in there that needs to be handled
#   SibSp | part of family size | number of siblings / spouses aboard the Titanic | combine in a single number
#   parch | part of family size | number of parents / children aboard the Titanic | combine in a single number
#   ticket | ID of ticket | scrub from list as it should not be correlated with survival
#   fare | Socio economic factor | amount paid. the amount is family based and needs to be normalized by family size
#   cabin | Socio economic factor | only first class cabins are listed with their deck. others are unknown.
#   embarked | Port of Embarkation | C = Cherbourg | Q = Queenstown | S = Southampton | might be correlated with fare
#Load from disk the raw training data
raw_data_train = lib_pandas.read_csv( "data/titanic_train.csv")

print("--------------------------------------------------------")
print("Process source data")
print( raw_data_train.describe() )

#process set, fill in the blanks and scrub unused columns
def compute_set_data( input_set ):
    # drop columns that are not going to be used by the model
    UNUSED_COLUMNS = ["Name", "Ticket", "Cabin", "Embarked"]
    result_set = input_set.drop(UNUSED_COLUMNS, axis=1)
    # Fare is computed by family. needs to be normalized by the number of family members
    result_set["Fare"] = result_set["Fare"]/(1+result_set["SibSp"]+result_set["Parch"])
    #Convert Sex to numeric value. 0.0 = female 1.0 = male
    result_set = result_set.replace( {"Sex" : {"female" : 0.0, "male": 1.0} } )
    #Family size. might be a correlation with family size, age and survival
    result_set["Family"] = 1+result_set["SibSp"]+result_set["Parch"]
    return result_set

# function to pre-process data
def compute_input_output(input_set):
    set_input = compute_set_data( input_set )
    # extract the survived column which serves as my output
    set_output = set_input.pop("Survived")
    return [set_input, set_output]

#Compute the training and validation data
[set_input_train, set_output_train] = compute_input_output(raw_data_train)

print("Data feeding the ML")
print("set_input_train\n", set_input_train )
print("set_output_train\n", set_output_train )

#Load from disk test data. Same as train data but lacks the survived column
raw_data_predict = lib_pandas.read_csv("data/titanic_test.csv")
#add a dummy survived column in which everybody dies (placeholder)
raw_data_predict["Survived"] = 0
#process the dataset
[set_input_predict, set_output_predict] = compute_input_output( raw_data_predict )
print("Prediction set. Needs to find out: Survived")
print(set_input_predict, set_output_predict)

#-------------------------------------------------------------
#	Neural network to fill unknown ages
#-------------------------------------------------------------
#	Use every train and test data that has age to compute a model that predicts age

print("---------------------------------------------------------")
print("Use a neural network to estimate unknown ages")

#Merge the training set and drop Survived column and every row which has a NaN (unknown ages)
age_training_data = lib_pandas.concat( (set_input_train, set_input_predict), axis = 0).dropna(axis = 0)
#obtain inputs and outputs of the age estimation ML model
x_train_age = age_training_data.drop("Age", axis=1)
y_train_age = age_training_data["Age"]
print("x_train_age\n", x_train_age, "\n", "y_train_age\n", y_train_age, "\n")

# create model
model_age_predictor = Sequential()
model_age_predictor.add(Dense(input_dim=x_train_age.shape[1], units=128, kernel_initializer="normal", bias_initializer="zeros"))
model_age_predictor.add(Activation("relu"))

for i in range(0, 8):
    model_age_predictor.add(Dense(units=64, kernel_initializer="normal", bias_initializer="zeros"))
    model_age_predictor.add(Activation("relu"))
    model_age_predictor.add(Dropout(.25))

model_age_predictor.add(Dense(units=1))
model_age_predictor.add(Activation("linear"))

model_age_predictor.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["accuracy"])

#train the model
model_age_predictor.fit(x_train_age.values, y_train_age.values, epochs=gi_age_estimate_epoch, verbose=2)

#expand the inputs to include rows with unknown ages
age_training_data = lib_pandas.concat( (set_input_train, set_input_predict), axis = 0)
x_train_age = age_training_data.drop("Age", axis=1)
y_train_age = age_training_data["Age"]
#Use the ML model to predict missing ages and compute error on known ages
age_training_data["New Age"] = model_age_predictor.predict(x_train_age.values)
age_training_data["Age Error"] = age_training_data["Age"] -age_training_data["New Age"]
print("mean age error: ", age_training_data["Age Error"].mean(), "\n" )
age_training_data.to_csv("data/age_predictor.csv")

#fill NaN age entries in training and prediction with the prediction value

#compute input whose age field is NaN
to_pred = set_input_train.loc[set_input_train["Age"].isnull()].drop(["Age"], axis=1)
#predict missing ages
p = model_age_predictor.predict(to_pred.values)
#boilerplate to fix shape
set_input_train["Age"].loc[set_input_train["Age"].isnull()] = p.reshape(set_input_train["Age"].loc[set_input_train["Age"].isnull()].shape)

to_pred = set_input_predict.loc[set_input_predict["Age"].isnull()].drop(["Age"], axis=1)
p = model_age_predictor.predict(to_pred.values)
set_input_predict["Age"].loc[set_input_predict["Age"].isnull()] = p.reshape(set_input_predict["Age"].loc[set_input_predict["Age"].isnull()].shape)

print(set_input_train)
print(set_input_predict)

#-------------------------------------------------------------
#	Neural network to fill unknown ages
#-------------------------------------------------------------

print("ML survived")
print("input shape", set_input_train.shape )
print("output shape", set_output_train.shape )
#Needs to be a 2D tensor with complementary outputs for the two categories
reshaped_set_output_train = lib_pandas.get_dummies( set_output_train)
print(reshaped_set_output_train.shape, "\n", reshaped_set_output_train)

# create model
model_survived_predictor = Sequential()
model_survived_predictor.add(Dense(input_dim=set_input_train.shape[1],  units=128, kernel_initializer="normal", bias_initializer="zeros"))
model_survived_predictor.add(Activation("relu"))

model_survived_predictor.add(Dropout(.05))
model_survived_predictor.add(Dense(units=256, kernel_initializer="random_normal", bias_initializer="zeros"))
model_survived_predictor.add(Activation("elu"))

for i in range(0, 8):
    model_survived_predictor.add(Dense(units=256, kernel_initializer="random_normal", bias_initializer="zeros"))
    model_survived_predictor.add(Activation("relu"))

model_survived_predictor.add(Dense(units=256, kernel_initializer="random_normal", bias_initializer="zeros"))
model_survived_predictor.add(Activation("elu"))
model_survived_predictor.add(Dropout(.05))

for i in range(0, 8):
    model_survived_predictor.add(Dense(units=128, kernel_initializer="random_normal", bias_initializer="zeros"))
    model_survived_predictor.add(Activation("relu"))

model_survived_predictor.add(Dense(units=2))
model_survived_predictor.add(Activation("softmax"))

model_survived_predictor.compile(loss="binary_crossentropy", optimizer=lib_tensorflow.keras.optimizers.RMSprop(learning_rate=0.0005, momentum=0.01 ), metrics=["accuracy"])
print( model_survived_predictor.summary() )
#train the model
model_survived_predictor.fit(set_input_train.values, reshaped_set_output_train.values, epochs=gi_survived_epoch, verbose=2)


#-------------------------------------------------------------
#	Export
#-------------------------------------------------------------

#Use NN to predict input
csv_training_prediction = set_input_train
p_survived = model_survived_predictor.predict_classes(set_input_train.values)
csv_training_prediction["Survived"] =  set_output_train
csv_training_prediction["Survived Prediction"] =p_survived
csv_training_prediction["Error"] = (csv_training_prediction["Survived"] != csv_training_prediction["Survived Prediction"])
csv_training_prediction.to_csv("data\model_test_prediction.csv", index=False)

print("training error rate: ", csv_training_prediction["Error"].mean() )

#Use Model to predict Survival of prediction set
#p_survived = lib_pandas.argmax(model_survived_predictor.predict(set_input_predict.values), axis=-1)
p_survived = model_survived_predictor.predict_classes(set_input_predict.values)
#Submission
submission = lib_pandas.DataFrame()
submission["PassengerId"] = set_input_predict["PassengerId"]
submission["Survived"] = p_survived
submission.to_csv("data\keras_submission.csv", index=False)

print("Survived prediction\n", submission, "\n")

print("training set survival rate: ", set_output_train.mean() )
print("prediction set survival rate: ", submission["Survived"].mean() )





