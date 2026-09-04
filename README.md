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
