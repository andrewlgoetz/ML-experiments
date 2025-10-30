import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../datasets/training_data.csv")

abalone = data.drop(data.columns[0], axis = 1)

features = abalone[["Length", "Diameter", "Height",
                    "Whole_weight", "Shucked_weight",
                    "Viscera_weight", "Shell_weight"]].values
#sepcify target
target = abalone["Rings"].values

# Polynomial Regression, using OLS
class poly_regression():
    def __init__(self,x_,y_) -> None:
        self.input = np.array(x_)
        self.target = np.array(y_)
    
    def preprocess(self,x_=None, y_=None):
        if x_ is None and y_ is None:
            x_ = self.input
            y_ = self.target
            
        numInstances = x_.shape[0] 
        numFeatures = x_.shape[1]

        ## create the X, adding columns iteratively
        x_train = [np.ones(numInstances)]  # ones
        for j in range(numFeatures):
                x_train.append(x_[:,j]) # all rows, column x
                x_train.append(x_[:, j]**2) # squared terms

        X = np.column_stack(x_train)

        #normalize the data
        for j in range(1, X.shape[1]):
            m = X[:, j].mean() # mean by column
            s = X[:, j].std()
            if  s == 0:
                s = 1.0 # avoid standard deviation of zero
            X[:, j] = (X[:, j] - m) / s
        
        # Y
        y_mean = y_.mean()
        y_std = y_.std()
        if y_std == 0:
            y_std = 1
        y_train = (y_ - y_mean)/y_std

        Y = (np.column_stack(y_train)).T

        return X, Y
    

    def train(self, X, Y):
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    def predict(self, X, beta):
        #predict using beta  
        return (X @ beta).ravel()

###### Training & Experimenting ######

## Seperate data into training and test
# np.random.seed(20)
# n = features.shape[0]
# perm = np.random.permutation(n)
# divide = int(0.8 * n)
# train_rows, test_rows = perm[:divide], perm[divide:]

# X_train = features[train_rows]
# Y_train = target[train_rows]
# X_test = features[test_rows]
# Y_test = target[test_rows]

## Hardcoding to use full dataset (not a train/test split):
X_train = features
Y_train = target
X_test = features
Y_test = target


## train the model (on train data)
model = poly_regression(X_train, Y_train) 
X, Y = model.preprocess()
beta = model.train(X,Y)


## predict on training data
yhat_train = model.predict(X, beta)
mse_train = np.mean((yhat_train - Y.ravel())**2)
rmse_train = np.sqrt(mse_train)
print("Training MSE:", mse_train)
print("Training RMSE:", rmse_train)


## predict on test data
X_test_processed, Y_test_processed = model.preprocess(X_test, Y_test)
yhat_test = model.predict(X_test_processed, beta)
mse_test = np.mean((yhat_test - Y_test_processed.ravel())**2)
rmse_test = np.sqrt(mse_test)
print("Test MSE:", mse_test)
print("Test RMSE:", rmse_test)


### PLOTTING ####


feature_names = ["Length","Diameter","Height",
                 "Whole_weight","Shucked_weight",
                 "Viscera_weight","Shell_weight"]


## Print beta values
beta_show = beta.ravel()
print("\nBeta values (x, x^2 for each feature):")
print("Bias: " + str(beta_show[0]))

for j, name in enumerate(feature_names): # columns are arranged x1, x1^2, x2, x2^2 ...
    beta_ = beta_show[1 + 2*j]
    beta_squared = beta_show[2 + 2*j]     
    print(name + ": linear term: " + str(beta_) + " squared_term: " + str(beta_squared))

## Show a plot for each feature ###

coefecients = beta.ravel()  # coefficients

for j, name in enumerate(feature_names):
    # Take the linear feature column
    x_feature = X[:, 1 + 2*j]
    y_target  = Y.ravel()   # Rings

    # Make a smooth range of x-values for the polynomial line
    x_line = np.linspace(x_feature.min(), x_feature.max(), 300)

    # Use the learned coefficients for this feature (bias, linear, squared)
    b0 = coefecients[0]
    b1 = coefecients[1 + 2*j]
    b2 = coefecients[2 + 2*j]

    # Compute the polynomial line: bias + b1*x + b2*x²
    y_line = b0 + b1*x_line + b2*(x_line**2)

    #### PLOT #####
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 6))

    # scatter actual data (normalized)
    ax.scatter(x_feature, y_target, label="Data (normalized)")

    # plot model slice
    ax.plot(x_line, y_line, color='r', linewidth=2, label="Model slice (deg 2)")

    # labels and title
    ax.set_xlabel(f"{name} (normalized)")
    ax.set_ylabel("Rings (normalized)")
    ax.set_title(f"Rings vs {name} — polynomial slice (others=0)")
    ax.legend()

    plt.show()