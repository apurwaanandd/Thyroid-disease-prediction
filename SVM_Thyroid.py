import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = "/Users/apurwaanand/Desktop/Datasets/Thyroid.txt"
df = pd.read_csv(file_path, delimiter=',')

# Show column names
print("🔍 Columns in dataset:", df.columns.tolist())

# Use 'classes' as the target
target_column = 'classes'

# Separate features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode the target variable if it's categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # Try 'linear', 'poly' etc.
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate
print("\n✅ Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# 📊 Visualization - Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 📊 Optional: Bar plot for Accuracy
accuracy = accuracy_score(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.bar(['SVM Accuracy'], [accuracy], color='skyblue')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.show()
