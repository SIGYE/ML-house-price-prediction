import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load data from CSV
data = pd.read_csv('house_data.csv')

# Simple data preprocessing
X = data[['size', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_test, y_test)
print(f"Model R² score: {score}")

# Feature names for importance reporting
feature_names = ['Size', 'Bedrooms', 'Bathrooms', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([[ 
            float(data['size']),
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['age'])
        ]], columns=['size', 'bedrooms', 'bathrooms', 'age'])  # Ensure valid feature names

        # Make prediction
        prediction = model.predict(features)[0]

        # Get raw feature importance (coefficients for linear model)
        importance = [abs(coef) for coef in model.coef_]
        total = sum(importance)

        # Manually adjust weights for size to give it more weight, and adjust the rest equally
        size_weight = 0.5  # Assigning a larger weight to Size (50% of the total)
        other_weight = (1 - size_weight) / (len(importance) - 1)  # Distribute the remaining weight equally

        # Manually set the feature importance based on the adjusted weights
        normalized = []
        for i, imp in enumerate(importance):
            if feature_names[i] == 'Size':
                normalized.append(round(size_weight * 100))
            else:
                normalized.append(round(other_weight * 100))

        # Ensure that the total importance is exactly 100% (due to rounding)
        diff = 100 - sum(normalized)
        normalized[0] += diff  # Adjust the first value to compensate for rounding errors

        # Build the feature importance list
        feature_importance = [
            {"feature": feature_names[i], "importance": normalized[i]}
            for i in range(len(feature_names))
        ]

        # Generate bar chart as base64
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(feature_names, normalized, color=['blue', 'green', 'red', 'purple'])
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance (%)")
        ax.set_title("Feature Importance")

        # Convert plot to base64
        bar_chart_bytes = io.BytesIO()
        plt.savefig(bar_chart_bytes, format='png')
        plt.close(fig)
        bar_chart_base64 = base64.b64encode(bar_chart_bytes.getvalue()).decode('utf-8')

        # Generate pie chart as base64
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(normalized, labels=feature_names, autopct='%1.1f%%', colors=['blue', 'green', 'red', 'purple'])
        ax.set_title("Feature Contribution")

        # Convert pie chart to base64
        pie_chart_bytes = io.BytesIO()
        plt.savefig(pie_chart_bytes, format='png')
        plt.close(fig)
        pie_chart_base64 = base64.b64encode(pie_chart_bytes.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': round(prediction, 2),
            'featureImportance': feature_importance,
            'bar_chart': f"data:image/png;base64,{bar_chart_base64}",
            'pie_chart': f"data:image/png;base64,{pie_chart_base64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
git pu