import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn import svm
x = [
    [ 7.79667589],
    [ 2.79825217],
    [ 2.06174503],
    [ 4.4713877 ],
    [ 7.20443649],
    [ 7.36014312],
    [ 4.70688117],
    [-0.40338389],
    [ 4.72266607],
    [ 1.20453709],
    [ 6.07593449],
    [ 7.69651292],
    [ 3.89733971],
    [ 4.7856351 ],
    [-0.59932188],
    [ 4.1507473 ],
    [ 0.04186784],
    [ 4.89562846],
    [ 2.38650347],
    [ 6.42758034]
]
y = [
    [ 318.28185696],
    [ 20.48143891],
    [ 11.97873995],
    [ 7.56902114],
    [ 224.15497306],
    [ 235.04403786],
    [ 17.75040067],
    [-107.86335911],
    [ 1.1140603 ],
    [ -7.67492972],
    [ 87.4263873 ],
    [ 293.22569099],
    [ -11.49557421],
    [ 6.4415876 ],
    [-152.88870565],
    [ -4.95755333],
    [ -79.53431819],
    [ 34.97246059],
    [ -4.50098315],
    [ 95.09276699]
]
x = np.array(x)
y = np.array(y)
plt.figure(figsize=(10, 8))
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot của x và y')
plt.show()
x_poly = PolynomialFeatures(degree=3).fit_transform(x)
reg = LinearRegression().fit(x_poly, y)
reg.coef_
reg.intercept_
x_interval = np.arange(np.min(x), np.max(x) + 0.1, 0.1).reshape(-1, 1)
x_interval_poly = PolynomialFeatures(degree=3).fit_transform(x_interval)
y_interval = reg.predict(x_interval_poly)

plt.figure(figsize=(10, 8))
plt.plot(x_interval, y_interval, 'r')
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kết quả hồi quy đa thức bậc 3')
plt.show()

##bài 2:
x = np.reshape([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50,
                4.00, 4.25, 4.50, 4.75, 5.00, 5.50], (-1, 1))
y = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
plt.figure(figsize=(10, 8))
color_map = ListedColormap(['b', 'r'])
label_map = ['Không mua', 'Có mua']
scatter = plt.scatter(x, y, c=y, cmap=color_map)
plt.xlabel('Giá nhà (tỷ VNĐ)')
plt.ylabel('Quyết định mua nhà')
plt.title('Scatter plot của x và y')
plt.legend(handles=scatter.legend_elements()[0], labels=label_map, title='Quyết định mua nhà')
plt.show()
reg = LogisticRegression().fit(x, y)
reg.coef_
reg.intercept_
x_interval = np.arange(np.min(x), np.max(x) + 0.1, 0.1).reshape(-1, 1)
y_interval = reg.predict_proba(x_interval)[:, 1]
threshold = -reg.intercept_.item() / reg.coef_.item()

plt.figure(figsize=(10, 8))
color_map = ListedColormap(['b', 'r'])
scatter = plt.scatter(x, y, c=y, cmap=color_map)
plt.plot(x_interval, y_interval, 'g')
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('Giá nhà (tỷ VNĐ)')
plt.ylabel('Quyết định mua nhà')
plt.title('Kết quả hồi quy logistic')
plt.show()