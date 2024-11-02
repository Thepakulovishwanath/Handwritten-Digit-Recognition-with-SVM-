from flask import Flask, render_template, request, redirect, url_for
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the digits dataset
digits = datasets.load_digits()

# Create Flask app
app = Flask(__name__)

# Directory for saving confusion matrix images
if not os.path.exists('static'):
    os.makedirs('static')

def train_and_evaluate_model():
    X = digits.data
    y = digits.target
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the SVM model with RBF kernel
    svm_clf = svm.SVC(kernel='rbf', gamma='auto')
    svm_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm_clf.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names, yticklabels=digits.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('static/confusion_matrix.png')  # Save confusion matrix as an image
    plt.close()  # Close the plot

    return accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    accuracy = train_and_evaluate_model()
    return render_template('results.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
