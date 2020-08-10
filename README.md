
# OneClassClassifier

Outlier detection using [One-class classifier](https://dongpin.data.blog/2020/07/30/anomaly-detection-with-occone-class-classification/). 

## Install
```python
python setup.py install
```


## Usage

```python
from occ.occ import *

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

### Implemented Models
```python
occ = occ()
occ.X = some_np_array
occ.train(model='SOMEMODEL')
```

 - deepsvdd : Deep-SVDD; [Deep One-Class Classification](http://proceedings.mlr.press/v80/ruff18a.html) [ICML2018; Ruff, Lukas *et al.,*] ; Tensorflow2 implementation
 - autoEncoder : AutoEncoder and reconstruction loss; [pyod](https://github.com/yzhao062/pyod) library
 - vae : Variational AutoEncoder and reconstruction loss; [pyod](https://github.com/yzhao062/pyod) library
 - isoForest : Isolation Forest; [Isolation Forest](https://ieeexplore.ieee.org/abstract/document/4781136/?casa_token=Cbf5YrMZKXcAAAAA:90G4z0yaa-0TbmIbDsQ0sPaj0oXOXWpevsK4PDn8YnV_EAL_yfOfxiYZo7xo2zKm5asJIDiovz0)[IEEE data mining conf 2008; Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.]; scikit-learn library
 - ocnn : One-Class Neural Networks; [Anomaly Detection Using One-Class Neural Networks](https://arxiv.org/abs/1802.06360) [Chalapathy, R., Menon, A. K., & Chawla, S.]; Tensorflow2 implementation
 - ocsvm : One-Class SVM; 
 [Support Vector Method for Novelty Detection] (http://papers.nips.cc/paper/1723-support-vector-method-for-novelty-detection.pdf) [NIPS2000; Schölkopf, Bernhard, *et al.*]; scikit-learn library
