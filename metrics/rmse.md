# Root mean square error (RMSE)

### 1. Metric Name and Description

RMSE calculates the square root of the average of squared differences between predicted and actual values. It gives us a measure of the typical size of the error in our predictions, expressed in the same units as the response variable.

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Where:

- $(n)$ is the number of observations
- $(y_i)$ is the actual observed value for the $(i)-th$ observation
- $(\hat{y}_i)$ is the predicted value for the $(i)-th$ observation

RMSE is always non-negative and a value of 0 would indicate a perfect fit to the data. The lower RMSE the better the fit. 

It can be challenging to compare RMSE values across different type of data with different scales. In order to make meaningful comparison you will need to normalize RMSE. 

### 2. Metric Behavior and Interpretation

RMSE is expressed in the same units as the target variable. For example, if you're predicting house prices in dollars, your RMSE will also be in dollars.

RMSE gives higher weight to large errors compared to small ones due to the squaring of errors before averaging. This makes it particularly sensitive to outliers.

Let's consider two scenarios with the same number of predictions and the same total error, but distributed differently:

#### Scenario A: Evenly distributed errors

- Actual values: [10, 20, 30, 40, 50]

- Predicted values: [12, 22, 32, 42, 52]

- Errors: [2, 2, 2, 2, 2]

````
xa = np.array([10, 20, 30, 40, 50])
ya_hat = np.array([12, 22, 32, 42, 52])
res_a = ya_hat - xa
print(res_a)
````
````
[2 2 2 2 2]
````

#### Scenario B: One large error, others small

- Actual values: [10, 20, 30, 40, 50]

- Predicted values: [10, 20, 30, 40, 58]

- Errors: [0, 0, 0, 0, 8]


````
xb = np.array([10, 20, 30, 40, 50])
yb_hat = np.array([10, 20, 30, 40, 58])
res_b = yb_hat - xb
print(res_b)
````
````
[0 0 0 0 8]
````

In Scenario A, we have evenly distributed errors, while in Scenario B, we have one large error and the rest are small. Despite both scenarios having the same total error of 10, the RMSE for Scenario B (3.58) is significantly higher than for Scenario A (2).

This difference occurs because RMSE squares the errors before averaging and taking the square root. Squaring gives more weight to larger errors.

````
mse_a = sum(res_a ** 2)/ len(res_a)
mse_b = sum(res_b ** 2)/ len(res_b)
print(f"MSE of Scenario A: {mse_a}")
print(f"MSE of Scenario B: {mse_b}")
````
````
MSE of Scenario A: 4.0
MSE of Scenario B: 12.8
````

````
print(f"RMSE of Scenario A: {np.sqrt(mse_a)}")
print(f"RMSE of Scenario B: {np.sqrt(mse_b)}")
````
````
RMSE of Scenario A: 2.0
RMSE of Scenario B: 3.5777087639996634
````

There's also no upper limit to RMSE. It can range from 0 to infinity.

Remember that RMSE is in the same units as your target variable. If you're predicting temperature in Celsius, an RMSE of 2 means your predictions are, on average, about 2°C off.

In general, a lower RMSE indicates better model performance. It suggests that, on average, your model's predictions are closer to the actual values.

The interpretation of what constitutes a "good" RMSE value depends on the context of your problem and the scale of your target variable. An RMSE of $10,000 might be excellent for house price predictions, but the same RMSE would be terrible for predicting a person's age.

It's often useful as starting point to interpret RMSE relative to the mean of your target variable. This expresses the error as a percentage of the average value. 

````
mean_a = np.mean(xa)
mean_b = np.mean(xb)
relative_rmse_a =  np.sqrt(mse_a) / mean_a
relative_rmse_b =  np.sqrt(mse_b) / mean_b
print(f"RMSE of Scenario A: {np.sqrt(mse_a)}")
print(f"RMSE of Scenario B: {np.sqrt(mse_b)}")
print(f"In Scenario A, on average, predictions deviate by about {relative_rmse_a*100:.2f}%")
print(f"In Scenario B, on average, predictions deviate by about {relative_rmse_b*100:.2f}%")
````
````
RMSE of Scenario A: 2.0
RMSE of Scenario B: 3.5777087639996634
In Scenario A, on average, predictions deviate by about 6.67%
In Scenario B, on average, predictions deviate by about 11.93%
````

Comparing RMSE to the standard deviation of your target variable can be informative. If RMSE < standard deviation, your model is performing better than always predicting the mean

````
rmse_a =  np.sqrt(mse_a) 
rmse_b =  np.sqrt(mse_b)
std_a = np.std(xa)
std_b = np.std(xb)
for s in ['A', 'B']:
    if rmse_a > std_a:
        print(f"For Scenario {s}, your model is performing worst than always predicting the mean")
    else:
        print(f"For Scenario {s}, your model is performing better than always predicting the mean")
````
````
For Scenario A, your model is performing better than always predicting the mean
For Scenario B, your model is performing better than always predicting the mean
````

You can also compare RMSE between different model implementations. If your RMSE is significantly lower than a simple baseline model, it suggests your model is adding value.

The benefit of these relative interpretations is that they provide context that absolute RMSE values lack. They help you answer questions like:

* Is my model's error small enough given the scale of what I'm predicting?

* How much have I improved over simpler approaches?

* Is my model competitive with what's currently considered good performance in my field?



### 3. Common Mistakes When Using the Metric

Some analysts or data scientists might choose RMSE without considering its sensitivity to outliers, which could lead to misleading results in certain situations.

````
# houses prices with outlier (400)
acts = np.array([210, 240, 185, 215, 400])
preds = np.array([200, 250, 180, 220,300])
res = preds-acts
mse = sum(res ** 2)/len(acts)
mae = sum(abs(res)) / len(acts)
rmse = np.sqrt(mse)
print(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}")
````
````
MAE: 26.00
RMSE: 45.28
````
````
# houses prices without outlier
acts = np.array([210, 240, 185, 215])
preds = np.array([200, 250, 180, 220])
res = preds-acts
mse = sum(res ** 2)/len(acts)
mae = sum(abs(res)) / len(acts)
rmse = np.sqrt(mse)
print(f"Removing the outliers\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")
````
````
Removing the outliers
MAE: 7.50
RMSE: 7.91
````

Imagine you're developing a predictive model for a ride-sharing company to estimate trip fares. Most trips are short to medium distance, with fares ranging from $10 to $50. Occasionally, there are long-distance trips with fares over $100. Emphasis on large errors, RMSE will heavily penalize large prediction errors, especially for those occasional long-distance trips. This could lead to the model being overly cautious with high-fare predictions to avoid large errors. The model might become overly sensitive to factors that influence long trips (like time of day or special events) to minimize these large errors.
This could potentially reduce accuracy for the more common short to medium trips.

Business impact could result in more accurate predictions for expensive trips but at the cost of slightly less accuracy for common trips.

MAE treats all prediction errors equally, regardless of trip distance or fare amount. This could result in a more balanced performance across all trip types. The model would likely aim for consistent performance across all fare ranges. It might not capture the nuances of factors affecting long-distance trips as well as RMSE would but, the business impact could lead to more consistent and predictable performance across all trip types.
It might result in slightly larger errors for long-distance trips, potentially leading to more customer dissatisfaction for these occasional but high-value rides.

### 4. Visual Aids

#### Scatter Plot: 

````
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
actual = np.random.rand(100) * 10
predicted = actual + (np.random.rand(100) - 0.5) * 2  # Add some noise

# Create the scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(actual, predicted, alpha=0.5)
plt.plot([0, 10], [0, 10], 'r--', lw=2)  # Add diagonal line

# Customize the plot
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values Scatter Plot')

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Set axis limits
plt.xlim(0, 10)
plt.ylim(0, 10)

# Add text annotation explaining RMSE
plt.text(0.5, 9, 'RMSE quantifies the differences\nbetween these points and the diagonal line', 
         fontsize=10, verticalalignment='top')

# Show the plot
plt.tight_layout()
plt.show()

# Calculate and print RMSE
rmse = np.sqrt(np.mean((predicted - actual)**2))
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
````

![Actual vs Predicted Scatter Plot](/images/rmse/actual_vs_predicted_scatter.png)


This shows actual vs predicted values. The differences between these points are what RMSE quantifies.

The distances between the points and the diagonal line represent the errors that RMSE quantifies. 

Correlation between actual and predicted values: The scatter of points shows how closely the predicted values align with the actual values. Points closer to the diagonal line indicate better predictions.

Over/under-predictions: Points above the diagonal line represent over-predictions, while points below represent under-predictions.

Prediction accuracy across different ranges: You can observe if the model performs consistently across all ranges of actual values or if it struggles in certain ranges

Outliers: Any points far from the diagonal line could indicate outliers or instances where the model's predictions are significantly off.

If there's a consistent pattern of points being above or below the line, it might indicate a bias in the model.


### 5. Competitions that Used This Metric


| Competition Name | Evaluation Metric | Link |
|------------------|-------------------|------|
| Zillow Prize: Zillow's Home Value Prediction (Zestimate) | log-error - mae | [Link](https://www.kaggle.com/competitions/zillow-prize-1/overview) |
| Rossmann Store Sales | RMSPE | [Link](https://www.kaggle.com/competitions/rossmann-store-sales/overview) |
| New York City Taxi Trip Duration | RMSLE | [Link](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview) |
| Playground Series - S3E21 | RMSE | [Link](https://www.kaggle.com/competitions/playground-series-s3e21/overview) |
| Playground Series - S3E6 | RMSE | [Link](https://www.kaggle.com/competitions/playground-series-s3e6) |
| Playground Series - S3E8 | RMSE | [Link](https://www.kaggle.com/competitions/playground-series-s3e8) |
| Playground Series - S3E11 | MAE | [Link](https://www.kaggle.com/competitions/playground-series-s3e11) |
| Playground Series - S3E14 | MAE | [Link](https://www.kaggle.com/competitions/playground-series-s3e14) |
| Playground Series - S3E16 | RMSE | [Link](https://www.kaggle.com/competitions/playground-series-s3e16) |
| Playground Series - S3E20 | RMSE | [Link](https://www.kaggle.com/competitions/playground-series-s3e20) |
| Playground Series - S3E1 | RMSE | [Link](https://www.kaggle.com/competitions/playground-series-s3e1) |


### 6. How to Optimize Models for This Metric

To optimize models for Root Mean Squared Error (RMSE), we need to understand its characteristics and how it differs from other metrics. RMSE is the square root of the mean squared error. It penalizes larger errors more heavily than smaller ones due to the squaring operation.

It's worth noting that using a squared error loss during training can align well with RMSE optimization.

Focus on features that help predict extreme values accurately

Create interaction terms that might help capture non-linear relationships

Apply transformations to reduce the impact of outliers.

Choose models that handle non-linearity well

Removing outliers can sometimes artificially improve your model's RMSE, but at the cost of losing potentially important information. In some cases, what appears to be an outlier might actually be a valid data point that your model should be able to handle. 

Winsorization is a technique where extreme values in your dataset are replaced with less extreme values. Typically, you'd set a percentile threshold (e.g., 5th and 95th percentiles) and replace any data points below or above these thresholds with the values at those percentiles. This preserves the overall distribution of your data while reducing the impact of extreme outliers. 

Let's say you have a feature with values: [1, 2, 3, 4, 100]. The outlier (100) would cause XGBoost to create splits and predictions that try to account for this extreme value.

After winsorization at the 90th percentile, your data might look like: [1, 2, 3, 4, 5]. Now XGBoost can create more balanced splits and predictions that work well for the majority of the data, while still preserving the information that there was a higher value in that last position.

Robust scaling scales your features using statistics that are robust to outliers. Instead of using mean and standard deviation (which are sensitive to outliers), robust scaling often uses median and interquartile range. This ensures that your scaling isn't overly influenced by extreme values.
