import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/apurwaanand/Desktop/Datasets/Thyroid.txt"
df = pd.read_csv(file_path, delimiter=',')

# Display the columns to check for the target column
print("🔍 Columns in dataset:", df.columns)

# Check for missing values
if df.isnull().sum().sum() > 0:
    print("⚠️ There are missing values in the dataset.")
else:
    print("✅ No missing values in the dataset.")

# Separate features and target (Change 'target' to actual column if necessary)
target_column = 'classes'  # Update this with the correct target column name
if target_column not in df.columns:
    raise ValueError(f"'{target_column}' not found in dataset columns.")
    
X = df.drop(target_column, axis=1)
y = df[target_column]

# Encode the target variable if it's categorical
le = LabelEncoder()  # Define le here
y_encoded = le.fit_transform(y)  # Fit and transform y

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize Gradient Boosting classifier
gb = GradientBoostingClassifier(random_state=42)

# Train the Gradient Boosting model
gb.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb.predict(X_test)

# Evaluate Gradient Boosting
print("\n✅ Gradient Boosting Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_gb)
print(cm)

print("\n✅ Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))
print("✅ Gradient Boosting Accuracy Score:", accuracy_score(y_test, y_pred_gb))

# Optional: Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Gradient Boosting Confusion Matrix')
plt.show()
