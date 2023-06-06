# Databricks notebook source
# MAGIC %md
# MAGIC # Keras TF 2.0 - Classification Project
# MAGIC
# MAGIC Explore a classification task with Keras API for TF 2.0
# MAGIC
# MAGIC ## The Data
# MAGIC
# MAGIC ### Breast cancer wisconsin (diagnostic) dataset
# MAGIC --------------------------------------------
# MAGIC
# MAGIC **Data Set Characteristics:**
# MAGIC
# MAGIC     :Number of Instances: 569
# MAGIC
# MAGIC     :Number of Attributes: 30 numeric, predictive attributes and the class
# MAGIC
# MAGIC     :Attribute Information:
# MAGIC         - radius (mean of distances from center to points on the perimeter)
# MAGIC         - texture (standard deviation of gray-scale values)
# MAGIC         - perimeter
# MAGIC         - area
# MAGIC         - smoothness (local variation in radius lengths)
# MAGIC         - compactness (perimeter^2 / area - 1.0)
# MAGIC         - concavity (severity of concave portions of the contour)
# MAGIC         - concave points (number of concave portions of the contour)
# MAGIC         - symmetry 
# MAGIC         - fractal dimension ("coastline approximation" - 1)
# MAGIC
# MAGIC         The mean, standard error, and "worst" or largest (mean of the three
# MAGIC         largest values) of these features were computed for each image,
# MAGIC         resulting in 30 features.  For instance, field 3 is Mean Radius, field
# MAGIC         13 is Radius SE, field 23 is Worst Radius.
# MAGIC
# MAGIC         - class:
# MAGIC                 - WDBC-Malignant
# MAGIC                 - WDBC-Benign
# MAGIC
# MAGIC     :Summary Statistics:
# MAGIC
# MAGIC     ===================================== ====== ======
# MAGIC                                            Min    Max
# MAGIC     ===================================== ====== ======
# MAGIC     radius (mean):                        6.981  28.11
# MAGIC     texture (mean):                       9.71   39.28
# MAGIC     perimeter (mean):                     43.79  188.5
# MAGIC     area (mean):                          143.5  2501.0
# MAGIC     smoothness (mean):                    0.053  0.163
# MAGIC     compactness (mean):                   0.019  0.345
# MAGIC     concavity (mean):                     0.0    0.427
# MAGIC     concave points (mean):                0.0    0.201
# MAGIC     symmetry (mean):                      0.106  0.304
# MAGIC     fractal dimension (mean):             0.05   0.097
# MAGIC     radius (standard error):              0.112  2.873
# MAGIC     texture (standard error):             0.36   4.885
# MAGIC     perimeter (standard error):           0.757  21.98
# MAGIC     area (standard error):                6.802  542.2
# MAGIC     smoothness (standard error):          0.002  0.031
# MAGIC     compactness (standard error):         0.002  0.135
# MAGIC     concavity (standard error):           0.0    0.396
# MAGIC     concave points (standard error):      0.0    0.053
# MAGIC     symmetry (standard error):            0.008  0.079
# MAGIC     fractal dimension (standard error):   0.001  0.03
# MAGIC     radius (worst):                       7.93   36.04
# MAGIC     texture (worst):                      12.02  49.54
# MAGIC     perimeter (worst):                    50.41  251.2
# MAGIC     area (worst):                         185.2  4254.0
# MAGIC     smoothness (worst):                   0.071  0.223
# MAGIC     compactness (worst):                  0.027  1.058
# MAGIC     concavity (worst):                    0.0    1.252
# MAGIC     concave points (worst):               0.0    0.291
# MAGIC     symmetry (worst):                     0.156  0.664
# MAGIC     fractal dimension (worst):            0.055  0.208
# MAGIC     ===================================== ====== ======
# MAGIC
# MAGIC     :Missing Attribute Values: None
# MAGIC
# MAGIC     :Class Distribution: 212 - Malignant, 357 - Benign
# MAGIC
# MAGIC     :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
# MAGIC
# MAGIC     :Donor: Nick Street
# MAGIC
# MAGIC     :Date: November, 1995
# MAGIC
# MAGIC This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
# MAGIC https://goo.gl/U2Uwz2
# MAGIC
# MAGIC Features are computed from a digitized image of a fine needle
# MAGIC aspirate (FNA) of a breast mass.  They describe
# MAGIC characteristics of the cell nuclei present in the image.
# MAGIC
# MAGIC Separating plane described above was obtained using
# MAGIC Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
# MAGIC Construction Via Linear Programming." Proceedings of the 4th
# MAGIC Midwest Artificial Intelligence and Cognitive Science Society,
# MAGIC pp. 97-101, 1992], a classification method which uses linear
# MAGIC programming to construct a decision tree.  Relevant features
# MAGIC were selected using an exhaustive search in the space of 1-4
# MAGIC features and 1-3 separating planes.
# MAGIC
# MAGIC The actual linear program used to obtain the separating plane
# MAGIC in the 3-dimensional space is that described in:
# MAGIC [K. P. Bennett and O. L. Mangasarian: "Robust Linear
# MAGIC Programming Discrimination of Two Linearly Inseparable Sets",
# MAGIC Optimization Methods and Software 1, 1992, 23-34].
# MAGIC
# MAGIC This database is also available through the UW CS ftp server:
# MAGIC
# MAGIC ftp ftp.cs.wisc.edu
# MAGIC cd math-prog/cpo-dataset/machine-learn/WDBC/
# MAGIC
# MAGIC .. topic:: References
# MAGIC
# MAGIC    - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
# MAGIC      for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
# MAGIC      Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
# MAGIC      San Jose, CA, 1993.
# MAGIC    - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
# MAGIC      prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
# MAGIC      July-August 1995.
# MAGIC    - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
# MAGIC      to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
# MAGIC      163-171.

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

# COMMAND ----------

df = pd.read_csv('/dbfs/FileStore/tables/hirak/Keras & Tensorflow Data/cancer_classification.csv')

# COMMAND ----------

df.head()

# COMMAND ----------

df.info()

# COMMAND ----------

df.describe().transpose()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA

# COMMAND ----------

sns.countplot(x='benign_0__mal_1',data=df)

# COMMAND ----------

plt.figure(figsize=(10,8))
sns.heatmap(df.corr())

# COMMAND ----------

df.corr()['benign_0__mal_1'].sort_values()

# COMMAND ----------

df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')

# COMMAND ----------

df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Test Split

# COMMAND ----------

X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

# COMMAND ----------

X_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Scaling Data

# COMMAND ----------

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating the Model
# MAGIC
# MAGIC     # For a binary classification problem
# MAGIC     model.compile(optimizer='rmsprop',
# MAGIC                   loss='binary_crossentropy',
# MAGIC                   metrics=['accuracy'])
# MAGIC                   
# MAGIC     

# COMMAND ----------

model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid')) #For binary classification, we use the 'sigmoid' activation function

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training the Model 
# MAGIC
# MAGIC ### Case 1: Choosing too many epochs and overfitting!

# COMMAND ----------

# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1)

# COMMAND ----------

model_loss = pd.DataFrame(model.history.history)
model_loss.head()

# COMMAND ----------

model_loss.plot() #Plot below indicates overfitting for validation loss - this need to be minimized, hence I resort to early stopping

# COMMAND ----------

# MAGIC %md
# MAGIC ##Early Stopping
# MAGIC
# MAGIC Case 2: Obviously trained too much! Will now use early stopping to track the val_loss and stop training once it begins increasing too much!

# COMMAND ----------

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# COMMAND ----------

# MAGIC %md
# MAGIC Stop training when a monitored quantity has stopped improving.
# MAGIC
# MAGIC     Arguments:
# MAGIC         monitor: Quantity to be monitored.
# MAGIC         min_delta: Minimum change in the monitored quantity
# MAGIC             to qualify as an improvement, i.e. an absolute
# MAGIC             change of less than min_delta, will count as no
# MAGIC             improvement.
# MAGIC         patience: Number of epochs with no improvement
# MAGIC             after which training will be stopped.
# MAGIC         verbose: verbosity mode.
# MAGIC         mode: One of `{"auto", "min", "max"}`. In `min` mode,
# MAGIC             training will stop when the quantity
# MAGIC             monitored has stopped decreasing; in `max`
# MAGIC             mode it will stop when the quantity
# MAGIC             monitored has stopped increasing; in `auto`
# MAGIC             mode, the direction is automatically inferred
# MAGIC             from the name of the monitored quantity.

# COMMAND ----------

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#Since we want to track validation loss, loss has to be 'minimized' and hence mode='min')

# COMMAND ----------

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop])

# COMMAND ----------

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding in DropOut Layers

# COMMAND ----------

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))       
#This 0.5 means that half the neurons in each batch will be turned off and hence their weights and biases won't be getting updated, 1 would mean all neurons will be turned off

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# COMMAND ----------

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop])

# COMMAND ----------

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Evaluation

# COMMAND ----------

#predictions = model.predict_classes(X_test).   #discontinued
predictions = (model.predict(X_test) > 0.5).astype("int32")
predictions

# COMMAND ----------

# https://en.wikipedia.org/wiki/Precision_and_recall
print(classification_report(y_test,predictions))

# COMMAND ----------

print(confusion_matrix(y_test,predictions))