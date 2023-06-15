from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

file_path = '/app/lung_cancer_examples.csv'
data = pd.read_csv(file_path)
columns_to_drop = ["Name", "Surname", "AreaQ"]
data = data.drop(columns=columns_to_drop)
X = data.drop('Result', axis=1)
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

@app.route('/')
def home():
    global train_accuracy, test_accuracy
    return render_template('accuracy.html', train_accuracy=train_accuracy, test_accuracy=test_accuracy)

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    # Get form input
    age = int(request.form['age'])
    smokes = int(request.form['smokes'])
    alkhol = int(request.form['alkhol'])

    # Use the trained model to make predictions
    user_data = pd.DataFrame({
        'Age': [age],
        'Smokes': [smokes],
        'Alkhol': [alkhol]
    })
    prediction = model.predict(user_data)[0]

    # Display the prediction result
    result = 'Positive' if prediction == 1 else 'Negative'
    return render_template('prediction.html', result=result)
@app.route('/performance')
def performance():
    return render_template('performance.html', results=results)
if __name__ == '__main__':
    app.run(port=5000)
