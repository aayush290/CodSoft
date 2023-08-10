import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

path = 'tested.csv'
data = pd.read_csv(path)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
X = data[features].copy()
y = data['Survived']

X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
X['Embarked'] = LabelEncoder().fit_transform(X['Embarked'].astype(str))


imputer = SimpleImputer(strategy='median')
X['Age'] = imputer.fit_transform(X['Age'].values.reshape(-1, 1))


X['Fare'] = imputer.fit_transform(X['Fare'].values.reshape(-1, 1))


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)


accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


survival_counts = pd.Series(y_pred).value_counts()
colors = ['red', 'green']
plt.bar(survival_counts.index, survival_counts.values, color=colors)
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.xlabel('Survival Outcome')
plt.ylabel('Count')
plt.title('Predicted Survival Distribution')
plt.show()