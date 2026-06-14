import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv(r"C:\Upendra\ML-MODEL-DEPLOYMENT-USING-FLASK-main (3) (1)\ML-MODEL-DEPLOYMENT-USING-FLASK-main\iris.csv")

print(df.head())

# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
X.describe()
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
y_pred_train=classifier.predict(X_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)
accuracy_score(y_test,y_pred)
accuracy_score(y_train,y_pred_train)
# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

import os
os.getcwd()
