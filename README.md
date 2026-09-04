# Machine Learning Algorithms

This repository contains simple implementations and examples of common machine
learning algorithms. The first example is a linear regression model that predicts
house prices from house size.

## Linear Regression

### Definition

Linear regression is a supervised machine learning algorithm used to model the
relationship between an input variable and a continuous output variable. It fits
the best straight line through the training data and uses that line to make
predictions.

In this project:

- **Feature (`x`)**: house size in square feet
- **Target (`y`)**: house price in lakhs
- **Prediction (`y_hat`)**: estimated price for a given house size

### Equation

The equation for a simple linear regression line is:

$$
\hat{y} = b_0 + b_1x
$$

Where:

- `y_hat` is the predicted output
- `b_0` is the intercept, or the predicted value when `x` is zero
- `b_1` is the slope, which represents the change in the prediction for each
	one-unit increase in `x`
- `x` is the input feature

The model calculates the slope and intercept using the averages of the input
and output values:

$$
b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}
{\sum (x_i - \bar{x})^2}
$$

$$
b_0 = \bar{y} - b_1\bar{x}
$$

Here, `x_bar` and `y_bar` are the mean values of `x` and `y`.

### Error Metrics

Error metrics measure how closely the model's predictions match the actual
target values. Lower values indicate better predictions.

#### Mean Squared Error (MSE)

Mean Squared Error is the average of the squared differences between the
actual values and the predicted values. Squaring the errors gives more weight
to larger errors.

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

#### Mean Absolute Error (MAE)

Mean Absolute Error is the average of the absolute differences between the
actual values and the predicted values. It represents the average prediction
error in the same units as the target.

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|
$$

Where:

- `n` is the number of observations
- `y_i` is the actual value
- `y_hat_i` is the predicted value

### Example Data

The example uses `linearregression/data.csv`:

| House size (sq ft) | Price (lakhs) |
| ------------------: | ------------: |
| 500                 | 20            |
| 800                 | 30            |
| 1000                | 40            |
| 1500                | 55            |
| 2000                | 70            |

### How to Run

Install the required packages:

```bash
pip install pandas matplotlib
```

Run the example from the repository root:

```bash
python linearregression/linerregression.py
```

When prompted, enter a house size in square feet. The program prints the
predicted price and displays a scatter plot with the fitted regression line.

### Limitations

- This example uses only one feature.
- The prediction assumes that the relationship between size and price is
	approximately linear.
- The dataset is very small and is intended for learning, not real-world price
	estimation.
