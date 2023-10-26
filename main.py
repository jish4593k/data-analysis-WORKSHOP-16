import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle

# Data Loading
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.info())
print('-' * 30)
print(train_data.describe())
print('-' * 30)
print(train_data.describe(include=['O']))
print('-' * 30)
print(train_data.head())
print('-' * 30)
print(train_data.tail())

# Data Cleaning
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()
train_features_encoded = encoder.fit_transform(train_features)
train_features_scaled = scaler.fit_transform(train_features_encoded)

# Decision Tree Construction
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(train_features_scaled, train_labels)


test_features_encoded = encoder.transform(test_features)
test_features_scaled = scaler.transform(test_features_encoded)

pred_labels = clf.predict(test_features_scaled)

# Calculate Decision Tree Accuracy
accuracy_decision_tree = round(clf.score(train_features_scaled, train_labels), 4)
print(f'Decision Tree Accuracy: {accuracy_decision_tree}')

# Data Visualization (Example: Age Distribution)
plt.figure(figsize=(8, 6))
sns.histplot(train_data['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

model = keras.Sequential([
    keras.layers.Input(shape=(train_features_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_features_scaled, train_labels, epochs=10, batch_size=32, validation_split=0.2)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init()
        self.fc1 = nn.Linear(train_features_scaled.shape[1], 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_features_scaled, train_labels = shuffle(train_features_scaled, train_labels)
train_features_tensor = torch.FloatTensor(train_features_scaled)
train_labels_tensor = torch.FloatTensor(train_labels.to_numpy()).view(-1, 1)

test_features_scaled_tensor = torch.FloatTensor(test_features_scaled)

train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / (len(train_loader)):.4f}")

# Evaluate the PyTorch model
model.eval()
with torch.no_grad():
    outputs = model(test_features_scaled_tensor)
    predicted = (outputs > 0.5).float()
    predicted = predicted.numpy().flatten()
    
# Convert predictions to integers (0 or 1)
predicted = predicted.astype(int)

test_data['Survived'] = predicted
test_data[['PassengerId', 'Survived']].to_csv('titanic_predictions.csv', index=False)
