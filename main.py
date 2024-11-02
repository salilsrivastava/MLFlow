# Installations
# 1. pip install mlflow
# 2. pip install psutil

# Steps to run
# 1. In terminal, run command -> mlflow ui --host 0.0.0.0 --port 5000
# 2. Right click on main.py and "run in interactive terminal"
# 3. Open localhost:5000 in browser and see the experimental results

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import psutil
import time
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the MLflow tracking URI to 'http'
mlflow.set_tracking_uri("http://localhost:5000")


def split_data(df):
    features = df[df.columns.drop(['HeartDisease', 'RestingBP', 'RestingECG'])].values
    target = df['HeartDisease'].values
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)
    return x_train, x_test, y_train, y_test

def preprocess_data(df):
    # Initialize the LabelEncoder
    le = LabelEncoder()
    # Create a deep copy of the DataFrame to avoid modifying the original data
    df1 = df.copy(deep=True)
    
    # Encode categorical features using LabelEncoder
    df1['Sex'] = le.fit_transform(df1['Sex'])
    df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
    df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
    df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
    df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])
    
    # Initialize scalers for normalization and standardization
    mms = MinMaxScaler()  # Normalization
    ss = StandardScaler()  # Standardization
    
    # Normalize and standardize selected features
    df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])
    df1['Age'] = ss.fit_transform(df1[['Age']])
    df1['RestingBP'] = ss.fit_transform(df1[['RestingBP']])
    df1['Cholesterol'] = ss.fit_transform(df1[['Cholesterol']])
    df1['MaxHR'] = ss.fit_transform(df1[['MaxHR']])
    
    # Return the processed DataFrame
    return df1

# Function for training the model
def train_model(X_train, y_train, max_depth=3, n_estimators=100):
    # Initialize the classifier
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    return clf


# Function to log model and system metrics to MLflow
def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # Log hyper parameters using in Random Forest Algorithm
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("n_estimators", model.n_estimators)

        # Log model metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        confusion = confusion_matrix(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
    
        print("Precision:", '{0:.2%}'.format(precision))
        print("Recall:", '{0:.2%}'.format(recall))
        print("F1 Score:", '{0:.2%}'.format(f1))
        
        # Log confusion matrix
        confusion_dict = {
            "true_positive": confusion[1][1],
            "false_positive": confusion[0][1],
            "true_negative": confusion[0][0],
            "false_negative": confusion[1][0]
        }
        mlflow.log_metrics(confusion_dict)

        # Log system metrics
        # Example: CPU and Memory Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        execution_time = {}  # Dictionary to store execution times for different stages
        # Example: Execution time for training the model
        start_time = time.time()
        model = train_model(X_train, y_train)
        end_time = time.time()
        execution_time["system_model_training"] = end_time - start_time

        # Log execution time 
        mlflow.log_metrics(execution_time)

        # Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")


def evaluate_model(classifier, x_test, y_test):
    predictions = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", '{0:.2%}'.format(accuracy))
    return accuracy

# Main function
def main():
    # Load the dataset
    data = pd.read_csv("heart.csv")  

    # Preprocess the data
    df_processed = preprocess_data(data)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = split_data(df_processed)

    # Train the model
    model = train_model(x_train, y_train)

    # Evaluate and log to MLflow
    log_to_mlflow(model, x_train,x_test, y_train, y_test)

if __name__ == "__main__":
    main()
