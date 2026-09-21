# 🌱 Indoor Plant Growth Prediction using Linear Regression

A beginner Machine Learning project that studies the relationship between daily light exposure and weekly indoor plant growth using **Simple Linear Regression**.

The project uses a synthetic dataset and implements the Linear Regression calculations using the **Ordinary Least Squares (OLS)** method.

## 📊 Dataset

The dataset contains two variables:

* `Light_Hours_Per_Day` — daily light exposure in hours
* `Weekly_Growth_cm` — weekly plant growth in centimeters

The dataset is synthetic and is used for learning and demonstrating Linear Regression.

## 🤖 Machine Learning Model

The project uses **Simple Linear Regression**:

$$
\hat{Y} = b_0 + b_1X
$$

Where:

* `b₁` = slope
* `b₀` = intercept
* `X` = input feature
* `Ŷ` = predicted value

The model parameters are calculated using the Ordinary Least Squares approach.

### Slope

$$
b_1 =
\frac{\sum(X-\bar X)(Y-\bar Y)}
{\sum(X-\bar X)^2}
$$

### Intercept

$$
b_0 = \bar Y - b_1\bar X
$$

The objective of OLS is to find the line that minimizes the **Sum of Squared Errors (SSE)**.

## 📈 Evaluation Metrics

The project calculates:

* R² Score
* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error

It also visualizes:

* Training data
* Residual errors
* Regression best-fit line

## 🖥️ Streamlit

A simple Streamlit interface is included to present the dataset, model explanation, visualizations, and model performance.

Run the application with:

```bash
streamlit run app.py
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit

## 📁 Project Structure

```text
indoor-plant-growth-linear-regression/
│
├── app.py
├── indoor_plant_growth.csv
├── requirements.txt
├── README.md
└── screenshots/
```

## 🎯 Learning Goals

This project was created to understand the fundamentals of Linear Regression, including:

* Train/test splitting
* OLS
* Slope and intercept
* Predictions
* Residuals
* SSE, MSE and RMSE
* R²
* Data visualization
* Basic Streamlit deployment
