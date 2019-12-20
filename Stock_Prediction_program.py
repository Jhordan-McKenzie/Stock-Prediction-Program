# This program predicts stock prices by using machine learning models.

# Install the dependencies
import quandl
quandl.ApiConfig.api_key = "4AEguTJzT-QpoKrsJqWy" #Had to set up an account with quandl to gain access to API key.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Get the stock data
df = quandl.get("WIKI/AMZN")
# Take a look at the data
print(df.head())

#Get the adjusted close price
df = df[['Adj. Close']]
#Take a look at the new data
print(df.head())


# A variable for predicting 'n' days out into the future
forecast_out = 30

# Create another column (the target or dependant variable) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

# Print the new data set
print(df.tail())

# Create the independant data set (x) #######
# Convert the dataframe to a numpy array
X = np.array(df.drop(['Prediction'], 1))

# Remove the last 'n' rows 
X = X[:-forecast_out]
print(X)


#Create the dependant data set (Y) #####
# Convert the dataframe to a numpy array (All of the values, including the NaN's)
Y = np.array(df['Prediction']) 
# Get all of the Y values, except the last 'n' rows
Y = Y[:-forecast_out]
print(Y)


# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# Create and train the Support Vector Machine (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# Create and train the Linear Regression Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)


# Set X_forecast equal to the last 30 rows of the original data set from the Adj. Close column
X_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
print(X_forecast)

# Print Linear Regression Model's prediction for the next 'n' days
lr_prediction = lr.predict(X_forecast)
print(lr_prediction)


# Print Support Vector Regressor Model's prediction for the next 'n' days
svm_prediction = svr_rbf.predict(X_forecast)
print(svm_prediction)
