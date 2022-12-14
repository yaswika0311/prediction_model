import pandas as pd
# pandas is used for data manipulation
import numpy as np
# It provides a multidimensional array object, as well as variations such as masks and matrices, which can be used for various math operations.
from sklearn import linear_model
# sklearn is used for ML
import seaborn as sns
# seaborn is used for statistical graphs
# DATA IN PANDAS
data = pd.read_csv("test.csv")
from sklearn.metrics import r2_score
np.random.randn(data.shape[0],1)
msk=np.random.rand(len(data))<=0.7
train=data[msk]
test=data[~msk]
X_train = train[["advexp"]]
Y_train = train["rev"]
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
X_test = test[['advexp']]
Y_test = reg.predict(X_test)
z=test['rev']
print(Y_test)
import matplotlib.pyplot as plt
# matplotlib is a plotting library for python
print(r2_score(z, Y_test) * 100, '%')
sns.regplot(x = test['advexp'], y = Y_test, data = data)
plt.show()

