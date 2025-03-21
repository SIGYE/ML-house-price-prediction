# ML-house-price-prediction
This is a Flask-based web application for predicting house prices based on input features like size, number of bedrooms, number of bathrooms, and age of the house. The application uses a linear regression model trained on historical house price data to make predictions and visualize feature importance.

## Features

- **Prediction**: Enter the size, number of bedrooms, number of bathrooms, and age of a house to predict its price.
- **Feature Importance**: Visualizes the importance of each feature (size, bedrooms, bathrooms, age) in determining the price of the house.
- **Bar Chart & Pie Chart**: Displays a bar chart and a pie chart showing the relative importance of each feature.

## Technologies Used

- **Flask**: For creating the web application.
- **scikit-learn**: For building and training the linear regression model.
- **Pandas**: For handling the dataset.
- **Matplotlib**: For visualizing feature importance.
- **NumPy**: For numerical operations.

## Setup

### Prerequisites

- Python 3.x
- `pip` (Python package manager)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
### 2. Install dependencies:
   pip install -r requirements.txt

### 3. Ensure you have a house_data.csv file that contains the data for training the model. This file should have the following columns:
    size: The size of the house in square feet (numeric).
    bedrooms: The number of bedrooms (numeric).
    bathrooms: The number of bathrooms (numeric).
    age: The age of the house in years (numeric).
    price: The price of the house (numeric).

### 4. Run the application:
    bash
    Copy
    Edit
    python app.py


### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
The project uses scikit-learn for machine learning.
The visualizations are created using Matplotlib.
vbnet
Copy
Edit
