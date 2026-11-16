
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\Users\aryan\Downloads\archive\WineQT.csv")

# First Five Records
print("First Five Records:")
print(df.head())

# Last Five Records
print("\nLast Five Records:")
print(df.tail())

# Random Five Records
print("\nRandom Five Records:")
print(df.sample(5))

# Number of Rows and Columns
print("\nDataset Shape:")
print(df.shape)

# Column Names
print("\nColumn Names:")
print(df.columns)

# Target Variable
print("\nTarget Variable: quality")

# Feature Information
print("\nDataset Information:")
print(df.info())

# Task 2: Dataset Inspection

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

# Task 3: Data Cleaning

print("\nBefore Cleaning:")
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Handle missing values
for column in df.select_dtypes(include=np.number).columns:
    df[column] = df[column].fillna(df[column].median())

print("\nAfter Cleaning:")
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

print("\nCleaned Dataset Shape:")
print(df.shape)

# Task 4: Exploratory Data Analysis (EDA)

# 1. Wine Quality Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.show()

# 2. Alcohol Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['alcohol'], kde=True)
plt.title("Alcohol Distribution")
plt.xlabel("Alcohol")
plt.show()

# 3. Volatile Acidity Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['volatile acidity'], kde=True)
plt.title("Volatile Acidity Distribution")
plt.show()

# 4. Fixed Acidity vs Wine Quality
plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='fixed acidity', data=df)
plt.title("Fixed Acidity vs Wine Quality")
plt.show()

# 5. Alcohol vs Wine Quality
plt.figure(figsize=(8, 5))
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title("Alcohol vs Wine Quality")
plt.show()
# Task 5: Correlation Analysis

# Calculate correlation
correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

# Create Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Wine Dataset Correlation Heatmap")
plt.show()

# Correlation with Wine Quality
print("\nCorrelation with Wine Quality:")
print(correlation['quality'].sort_values(ascending=False))

# Task 6: Feature Selection and Target Separation

# Remove unnecessary Id column
df = df.drop('Id', axis=1)

# Separate Features and Target
X = df.drop('quality', axis=1)
y = df['quality']

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeature Columns:")
print(X.columns)
# Task 7: Train-Test Split

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)
# Task 8: Feature Scaling

from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

print("\nOriginal Training Data:")
print(X_train.head())

print("\nScaled Training Data:")
print(X_train_scaled[:5])

print("\nScaled Testing Data:")
print(X_test_scaled[:5])

print("\nTraining Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)
# Task 9: Logistic Regression Model

from sklearn.linear_model import LogisticRegression

# Create the model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)

print("\nLogistic Regression Model Trained Successfully!")

print("\nActual Values:")
print(y_test.head(10).values)

print("\nPredicted Values:")
print(y_pred_lr[:10])


# Task 11: Support Vector Machine

from sklearn.svm import SVC

svm_model = SVC(kernel='rbf')

svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

print("SVM Model Trained Successfully!")
print("Predictions:", y_pred_svm[:10])
# Task 13: Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    "Logistic Regression": y_pred_lr,
    "SVM": y_pred_svm
}

for name, predictions in models.items():

    print("\nModel:", name)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("Precision:", precision_score(y_test, predictions, average='weighted', zero_division=0))

    print("Recall:", recall_score(y_test, predictions, average='weighted', zero_division=0))

    print("F1 Score:", f1_score(y_test, predictions, average='weighted', zero_division=0))

    # Task 14: Confusion Matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_svm)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(confusion_matrix=cm)

display.plot(cmap='Blues')

plt.title("SVM Confusion Matrix")

plt.show()
# Task 17: Wine Quality Prediction

print("WINE QUALITY PREDICTION")

user_data = []

for feature in X.columns:
    value = float(input("Enter " + feature + ": "))
    user_data.append(value)

user_df = pd.DataFrame([user_data], columns=X.columns)

user_scaled = scaler.transform(user_df)

prediction = svm_model.predict(user_scaled)

quality = int(prediction[0])

print("\nPredicted Wine Quality:", quality)

if quality <= 4:
    print("Category: Low Quality")

elif quality <= 6:
    print("Category: Medium Quality")

else:
    print("Category: High Quality")
    # Task 18: Final Analysis and Conclusion

print("\nWINE QUALITY PREDICTION PROJECT")

print("\nDataset Shape:", df.shape)

print("Total Features:", X.shape[1])

print("Training Records:", X_train.shape[0])

print("Testing Records:", X_test.shape[0])

print("\nModel Comparison:")

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred_lr))

print("SVM Accuracy:",
      accuracy_score(y_test, y_pred_svm))

print("\nCONCLUSION:")

print("Wine quality prediction was performed using Machine Learning.")

print("Logistic Regression and SVM models were trained.")

print("Both models were evaluated using performance metrics.")

print("The project predicts wine quality based on its chemical properties.")

print("\nPROJECT COMPLETED!")