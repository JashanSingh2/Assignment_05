import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Data Preprocessing and Feature Engineering
# Handle missing values
num_cols = train.select_dtypes(include=[np.number]).columns
train[num_cols] = train[num_cols].fillna(train[num_cols].median())

# Handle test data - exclude SalePrice column if it exists
test_num_cols = test.select_dtypes(include=[np.number]).columns
test[test_num_cols] = test[test_num_cols].fillna(train[test_num_cols].median())

cat_cols = train.select_dtypes(include=['object']).columns
for col in cat_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    if col in test.columns:
        test[col] = test[col].fillna(train[col].mode()[0])

# One-hot encode categorical features
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Align train and test data
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Target and Features
X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Simple Linear Regression (1 Feature)
if 'GrLivArea' in train.columns:
    X1 = train[['GrLivArea']]
    y1 = y
    simple_lr = LinearRegression().fit(X1, y1)
    pred_simple = simple_lr.predict(X1)
    print("Simple Linear R2:", r2_score(y1, pred_simple))
else:
    print("Feature 'GrLivArea' not found for simple linear regression.")

# Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)
print("Multiple Linear RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_mlr)))

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
poly_model = LinearRegression().fit(X_poly_train, y_train)
y_pred_poly = poly_model.predict(X_poly_test)
print("Polynomial RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# Lasso Regression
lasso = Lasso(alpha=5.0, max_iter=20000, tol=1e-6)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Lasso RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

# ElasticNet Regression
elastic = ElasticNet(alpha=5.0, l1_ratio=0.5, max_iter=20000, tol=1e-6)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
print("ElasticNet RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_elastic)))

# Classification - Bin SalePrice
bins = [0, 150000, 250000, 600000]
labels = ['Low', 'Medium', 'High']
y_class = pd.cut(y, bins=bins, labels=labels)

# Drop rows where y_class is NaN
mask = ~y_class.isna()
X_class = X_scaled[mask]
y_class_clean = y_class[mask]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_class, y_class_clean, test_size=0.2, random_state=42)

# Logistic Regression
clf_lr = LogisticRegression(max_iter=1000)
clf_lr.fit(X_train_clf, y_train_clf)
pred_lr = clf_lr.predict(X_test_clf)
print("Logistic Regression Accuracy:", accuracy_score(y_test_clf, pred_lr))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_clf, y_train_clf)
pred_nb = nb.predict(X_test_clf)
print("Naive Bayes Accuracy:", accuracy_score(y_test_clf, pred_nb))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_clf, y_train_clf)
pred_knn = knn.predict(X_test_clf)
print("KNN Accuracy:", accuracy_score(y_test_clf, pred_knn))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_clf, y_train_clf)
pred_dt = dt.predict(X_test_clf)
print("Decision Tree Accuracy:", accuracy_score(y_test_clf, pred_dt))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_clf, y_train_clf)
pred_rf = rf.predict(X_test_clf)
print("Random Forest Accuracy:", accuracy_score(y_test_clf, pred_rf))

# SVM
svm = SVC()
svm.fit(X_train_clf, y_train_clf)
pred_svm = svm.predict(X_test_clf)
print("SVM Accuracy:", accuracy_score(y_test_clf, pred_svm))

# End of script 