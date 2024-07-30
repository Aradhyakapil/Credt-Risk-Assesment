import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ['Status', 'Duration', 'Credit_history', 'Purpose', 'Credit_amount', 'Savings',
           'Employment', 'Installment_rate', 'Personal_status', 'Other_parties',
           'Residence_since', 'Property_magnitude', 'Age', 'Other_payment_plans',
           'Housing', 'Existing_credits', 'Job', 'Num_dependents', 'Own_telephone',
           'Foreign_worker', 'Class']

data = pd.read_csv(url, delimiter=' ', header=None, names=columns)

# Preprocess the data
# Map target variable 'Class' to binary (1 for good credit, 0 for bad credit)
data['Class'] = data['Class'].map({1: 1, 2: 0})

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Define feature variables (X) and the target variable (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
