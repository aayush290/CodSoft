import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

dataset_path = 'Iris.xlsx'
data = pd.read_excel(dataset_path)

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_names = label_encoder.inverse_transform(y_pred)
y_test_names = label_encoder.inverse_transform(y_test)

print(classification_report(y_test_names, y_pred_names))
