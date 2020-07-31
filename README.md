# OneClassClassifier

Outlier detection using [One-class classifier](https://dongpin.data.blog/2020/07/30/anomaly-detection-with-occone-class-classification/). 



## Usage

```python
from occ import *

## Load data
ocsvm = occ()
ocsvm.load_data_mat(file)
"""
Or
ocsvm.load_data_csv(file)
Or
ocsvm.X = some_np_array ##(n_samples, n_features)
"""

## Train the model and get results
ocsvm.train(model='ocsvm', kernel='rbf', norm=True) # If norm=True, data will be normalized(L2)
Y_scores = ocsvm.get_score(norm=True)
Y_hat = ocsvm.predict(norm=True)

## Visualization of the results
occ.show_projection(ocsvm.X, Y_scores, title="Score ocsvm with rbf kernel", markersize=100)
occ.show_projection(ocsvm.X, Y_hat, title="Prediction ocsvm with rbf kernel", markersize=100)

## Export the result score and outliers
ocsvm.export_outliers("outliers.csv", Y_hat) ## Raw data of outliers
ocsvm.export_csv("scores.csv", Y_scores) ## Raw data with scores
ocsvm.export_csv("predictions.csv", Y_hat) ## Raw data with predictions
```



### Adjusting "nu" as the prior knowledge of the proportion of outliers

```python
few_outliers = ocsvm.train(model='ocsvm', nu=0.01) ## 1% outliers assumed
many_outliers = ocsvm.train(model='ocsvm', nu=0.2) ## 20% outliers assumed
```



### Using small subset for training

```python
ocsvm = occ()
ocsvm.X = some_np_array
ocsvm.train(model='isolationForest', sampling=0.1) ## Only 10% of data will be used for training
```



