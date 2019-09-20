import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from warnings import filterwarnings
import tensorflow as tf
import keras.backend as K
K.clear_session()

filterwarnings('ignore')

cancer = load_breast_cancer()
cancer.keys()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df2 = pd.DataFrame(data=cancer.target)
X = df.values
y = df2.values
sns.pairplot(df) # Pairplot for Exploratory Data Analysis
sns.jointplot(df.loc[:,'worst concavity'], df.loc[:,'worst concave points'], kind="regg", color="#ce1414")
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f')
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis') # Heatmap to check if there is any missing Data
plt.tight_layout()

from sklearn.model_selection import train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=30 ))

model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=16, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=10)
model.save('ANN_Cancer.h5')

y_pred = model.predict(X_val)
y_pred = (y_pred>0.5)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, cmap='coolwarm')

# Checking it out on a single data entry

trial = df.values
print(trial[0])
prediction = model.predict(sc.transform(np.array([[1.799e+01 ,1.038e+01 ,1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01, 3.001e-01,
                                                   1.471e-01, 2.419e-01, 7.871e-02,1.095e+00,9.053e-01, 8.589e+00, 1.534e+02,
                                                   6.399e-03, 4.904e-02, 5.373e-02,1.587e-02,3.003e-02, 6.193e-03, 2.538e+01,
                                                   1.733e+01, 1.846e+02, 2.019e+03,1.622e-01,6.656e-01, 7.119e-01, 2.654e-01,
                                                   4.601e-01, 1.189e-01]])))
prediction = (prediction > 0.5)
print(prediction)
print(trial[1])
prediction2 = model.predict(sc.transform(np.array([[2.057e+01, 1.777e+01 ,1.329e+02, 1.326e+03, 8.474e-02 ,7.864e-02 ,8.690e-02,
                                                    7.017e-02, 1.812e-01, 5.667e-02,5.435e-01 ,7.339e-01,3.398e+00, 7.408e+01,
                                                    5.225e-03, 1.308e-02, 1.860e-02,1.340e-02 ,1.389e-02,3.532e-03, 2.499e+01,
                                                    2.341e+01, 1.588e+02, 1.956e+03,1.238e-01 ,1.866e-01,2.416e-01, 1.860e-01,
                                                    2.750e-01, 8.902e-02]])))
prediction2 = (prediction2 >0.5)
print(prediction2)
print(trial[19])
prediction3 = model.predict(sc.transform(np.array([[1.354e+01, 1.436e+01, 8.746e+01, 5.663e+02, 9.779e-02 ,8.129e-02, 6.664e-02,
                                                    4.781e-02,1.885e-01,5.766e-02,2.699e-01 ,7.886e-01, 2.058e+00, 2.356e+01,
                                                    8.462e-03,1.460e-02,2.387e-02,1.315e-02 ,1.980e-02, 2.300e-03, 1.511e+01,
                                                    1.926e+01,9.970e+01,7.112e+02,1.440e-01 ,1.773e-01, 2.390e-01, 1.288e-01,
                                                    2.977e-01,7.259e-02]])))
prediction3 = (prediction3 > 0.5)
print(prediction3)

loss = model.evaluate(X_test, y_test, verbose=1, batch_size=30)


# Creating a Classifier Random Forest
from sklearn.ensemble import RandomForestClassifier
err_rate = []
for i in range(1,50):
    classifier = RandomForestClassifier(n_estimators=i)
    classifier.fit(X_train, y_train)
    y_pred_i = classifier.predict(X_test)
    err_rate.append(np.mean(y_pred_i!=y_test))
plt.title("Elbow Method")
plt.plot(range(1,50),err_rate)
plt.show()


classifier = RandomForestClassifier(n_estimators=5,n_jobs=-1)
classifier.fit(X_train, y_train)
RandomForestpred = classifier.predict(X_val)

print(classification_report(RandomForestpred, y_val))
print(confusion_matrix(RandomForestpred, y_val))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
ll = LogisticRegression()
ll.fit(X_train, y_train)
llp = ll.predict(X_val)

print(classification_report(llp, y_val))
print(confusion_matrix(llp, y_val))

