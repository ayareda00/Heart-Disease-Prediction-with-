import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:\\Users\\ayare\\OneDrive\\Desktop\\heartv1.csv")

data.head()

data.tail()

"""# 2. Exploraroty Data Analysis


*   Data Analysis


"""

data.info()

data.describe()

len(data)

len(data.columns)

data.shape

# Check for null values in the DataFrame
null_values = data.isnull().sum()
print(null_values)

#check for the number of duplicate rows
duplicate_rows = data.duplicated().sum()
print(duplicate_rows)

sex_counts = data['sex'].value_counts()
print(sex_counts)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

data_cleaned = data.drop_duplicates()

print(f"Shape before removing duplicates: {data.shape}")
print(f"Shape after removing duplicates: {data_cleaned.shape}")

sex_counts = data_cleaned['sex'].value_counts()
print(sex_counts)

# Convert the 'sex' column to numerical data
data_cleaned['sex'] = LabelEncoder().fit_transform(data_cleaned['sex'])
data_cleaned['sex'].head()

X = data_cleaned.drop('target', axis=1)
y = data_cleaned['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)

""" *the training data = 80% from the total data*

 *the testing data = 20% from the total data*
"""

plt.figure(figsize=(12, 8))
sns.countplot(x='sex', data=data_cleaned, palette='Pastel1')
plt.xticks(rotation=90)

data_cleaned.hist(bins=20, figsize=(15, 12))
plt.show

plt.figure(figsize=(25 , 15))
sns.boxplot(data=data_cleaned, palette= 'Pastel1')
plt.show

#correlatian matrix
numerical_data = data_cleaned.select_dtypes(include=[np.number])
corr_matrix = numerical_data.corr()
corr_matrix['age'].sort_values(ascending=False)

plt.figure(figsize=(30,20))
sns.heatmap(corr_matrix, annot=True, cmap='Pastel1')

"""#plotting features"""

plt.figure(figsize=(10,5))
sns.scatterplot(x='age', y='Heart Disease Risk Score', data=data_cleaned ,hue='cp')

sns.pairplot(data_cleaned)

"""# 3. Choosing and Training model

"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

"""1- Logistic Regression Model"""

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""2- Decision Tree Model"""

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""3- Random Forest Model"""

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""4- Gradient Boosting Model"""

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""5- K-Nearest Neighbors (KNN) Model"""

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""6- Support Vector Machine (SVM) Model"""

svm_model = SVC(kernel='linear', random_state=42)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""7- Gaussian Naive Bayes Model"""

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""8- CatBoost Classifier Model"""

catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, cat_features=[], verbose=200)

catboost_model.fit(X_train, y_train)

y_pred = catboost_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""9- Stochastic Gradient Descent (SGD) Classifier Model"""

sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)

sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred)*100 ,'%')
print('Classification Report:\n', classification_report(y_test, y_pred))

"""#  Save the trained models"""

import pickle

with open('log_reg_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)
with open('dt_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(nb_model, f)
with open('catboost_model.pkl', 'wb') as f:
    pickle.dump(catboost_model, f)
with open('sgd_model.pkl', 'wb') as f:
    pickle.dump(sgd_model, f)

import joblib

joblib.dump(log_reg, 'log_reg_model.joblib')
joblib.dump(dt_model, 'dt_model.joblib')
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(gb_model, 'gb_model.joblib')
joblib.dump(knn_model, 'knn_model.joblib')
joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(nb_model, 'nb_model.joblib')
joblib.dump(catboost_model, 'catboost_model.joblib')
joblib.dump(sgd_model, 'sgd_model.joblib')

import joblib

# Replace rf_model with the model you want to save
joblib.dump(rf_model,'rf_model.pkl')

from sklearn.preprocessing import StandardScaler
import joblib

# Assume X_train has been defined during model training
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler,'scaler.pkl')
