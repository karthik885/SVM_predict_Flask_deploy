from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")  # Ensure it looks in 'templates' folder
CORS(app)  # Enable CORS for frontend requests

# Sample dataset
data = {
    "feature1": [2.5, 3.6, 1.8, 3.1, 3.0, 2.1, 1.5, 3.2],
    "feature2": [1.5, 2.1, 1.8, 2.2, 2.0, 1.9, 1.4, 2.1],
    "target": [0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Train the SVM model
X = df[["feature1", "feature2"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

# Route for frontend
@app.route("/")
def index():
    return render_template("index.html")  # Ensure index.html is in "templates" folder

# API Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        feature1 = float(data.get("feature1"))
        feature2 = float(data.get("feature2"))

        prediction = model.predict(np.array([[feature1, feature2]]))[0]

        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000,debug=True)
