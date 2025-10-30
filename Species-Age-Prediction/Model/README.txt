To run the code:
    execute `python <filename>`


To specify an input csv file:
    modify the line that says `data =` labeled "#import data" at the top of either file


A polynomial regression model of degree 2 using the ordinary least squares (OLS) method, due to its improved accuracy over linear Gradient Descent, and the nature/shape of data, the relationship of which seemed better captured using a degree 2 polynomial. I did not use interaction terms.

To evaluate performance, I calculated mean squared error (MSE) and root mean squared error (RMSE) on both the training and test data, consistent with the assignment’s requirement to report loss.

For visualization, I plotted each feature individually against the target variable, but kept the model trained on all seven features (ie set the others to zero). This shows the interesting dynamics of each feature in system of many dimensions (especially at the ends of the data, sometimes going in negative correlating trends).

I reported the beta coefficients for each feature.





ref:
    numpy.org/devdocs (numpy reference)

