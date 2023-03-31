import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from imblearn.over_sampling import SMOTE

data = pd.read_csv('framingham.csv')

data = data.dropna()

y = data['TenYearCHD']
X = data.drop('TenYearCHD', axis=1)
X = X.drop('education', axis=1)

# Se realizó un oversampling debido al desbalanceo de datos.
oversampler = SMOTE(random_state=0)

X, y = oversampler.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

poly = PolynomialFeatures(degree=2)
X_train = poly.fit_transform(X_train)
X_test = poly.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(accuracy_score(prediction, y_test))