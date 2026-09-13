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

## Polynomial Regression

### Definition

Polynomial regression is a supervised machine learning algorithm used when the
relationship between the input variable and the target is nonlinear. Instead of
fitting a straight line, it fits a polynomial curve to the data.

In this project:

- **Feature (`x`)**: hours studied
- **Target (`y`)**: exam score
- **Prediction (`y_hat`)**: estimated exam score for a given number of study
  hours

### Equation

A second-degree polynomial regression model is represented as:

$$
\hat{y} = b_0 + b_1x + b_2x^2
$$

Where:

- `y_hat` is the predicted value
- `b_0` is the intercept
- `b_1` is the coefficient for the linear term
- `b_2` is the coefficient for the squared term
- `x` is the input feature

This project computes the coefficients using a least-squares approach with a
design matrix:

$$
M = \begin{bmatrix}
1 & x_1 & x_1^2 \\
1 & x_2 & x_2^2 \\
\vdots & \vdots & \vdots \\
1 & x_n & x_n^2
\end{bmatrix}
$$

and solves:

$$
\beta = (M^TM)^{-1}M^Ty
$$

### Example Data

The example uses `polynomialregression/data.csv`:

| Hours studied | Exam score |
| -------------: | ---------: |
| 1              | 35         |
| 2              | 40         |
| 3              | 55         |
| 4              | 70         |
| 5              | 80         |
| 6              | 90         |

### How to Run

Run the example from the repository root:

```bash
python polynomialregression/polynomialregression.py
```

When prompted, enter the number of study hours. The program prints the
estimated exam score and displays a scatter plot of the original data with the
polynomial curve fitted to it.

### Limitations

- This example uses a simple quadratic model.
- More complex relationships may require a higher-degree polynomial or a
  different algorithm.
- The dataset is small and intended for learning rather than production-grade
  prediction.
