import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset (assuming 'classification' is the target column)

file_path = "/Users/apurwaanand/Desktop/Datasets/Thyroid.txt"
df = pd.read_csv(file_path, delimiter=',')

# Drop rows with missing values
df.dropna(inplace=True)

# Label encoding for categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Assuming 'classification' is the target column
X = df.drop(columns=['classes'])
y = df['classes']

# Split dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=9)
knn_classifier.fit(X_train, y_train)

# Make predictions on test set
y_pred = knn_classifier.predict(X_test)

# Evaluate model
accuracy = knn_classifier.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}\n')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1], ['No Disease', 'Disease'])
plt.yticks([0, 1], ['No Disease', 'Disease'])
plt.show()
