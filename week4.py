# Week4 Test
# This is a simple real estate price prediction problem
# Source URL: https://www.kaggle.com/quantbruce/real-estate-price-prediction
# Data Set Characteristic
# Target: Column 11 is a quantitative measure of house price of unit area
# Attribute Information:
# 1   No
# 2   X1 transaction date
# 3   X2 house age
# 4   X3 distance to the nearest MRT station
# 5   X4 number of convenience stores
# 6   X5 latitude
# 7   X6 longitude

# feel free to plot any graph for data visualization but it will not be graded



# import the library
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Question1
def load():
	'''
	load the dataset
	return df
	'''
	df = pd.read_csv('https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction/download')
	#SOLUTION START( ~ 1 line of code)
	

	#SOLUTION END
	return df

df = load()
print(df.head())


# Since there is no need of 'No' column and 'transcation date',
# so we will drop them
df.drop(['No'],axis='columns', inplace=True)
df.drop(['X1 transaction date'], axis='columns' ,inplace=True)


#Question2
def getShape():
	'''
	return the shape of dataframe
	'''
	#SOLUTION START( ~ 1 line of code)
	
	return df.shape
	#SOLUTION END

print(getShape())

#Question3
def getInfo():
	'''
	return the information of information about a DataFrame
	including the index dtype and columns, non-null values and memory usage.
	'''
	#SOLUTION START( ~ 1 line of code)
	
	return df.info()
	#SOLUTION END

print(getInfo())

# Question4
def checkNull():
	'''
	this function check if any null value
	expected output:
	No                                        0
	X1 transaction date                       0
	X2 house age                              0
	X3 distance to the nearest MRT station    0
	X4 number of convenience stores           0
	X5 latitude                               0
	X6 longitude                              0
	Y house price of unit area                0
	dtype: int64
	'''

	#SOLUTION START( ~ 1 line of code)
	return df.isnull().sum()

	#SOLUTION END


print(checkNull())

#Question5
def getStatistic():
	'''
	return the statistic of Dataframe such as count, mean, std, min, max, etc
	'''

	#SOLUTION START( ~ 1 line of code)
	return df.describe()
	#SOLUTION END

print(getStatistic())


#Now, we will extract the target variable which is 'Y house price of unit area' from the dataframe for training

df_y = df[['Y house price of unit area']]
df_X = df.drop(['Y house price of unit area'],axis='columns')

print(df_y.head())
print(df_X.head())


#Question 6
# Split the data into 80%training/20%testing sets
# IMPORTANT:for grading purpose,  please set the same seed random_state = 42

def split():
    '''
    IMPORTANT: this function will return four values X_train, X_test, y_train, y_test
    '''
    # SOLUTION START
    X = df.drop(columns=['Y house price of unit area'])  # Features
    y = df['Y house price of unit area']  # Target variable
    return train_test_split(X, y, test_size=0.2, random_state=42)
    # SOLUTION END

X_train, X_test, y_train, y_test = split()


def makeModel():
    '''
    This function will create a linear regression object, train the model using the training sets,
    make predictions using the testing set, and return coefficients, intercept, mse, rmse, r2score.
    '''

    # Create linear regression object
    # SOLUTION START
    model = LinearRegression()
    # SOLUTION END

    # Train the model using the training sets
    # SOLUTION START
    model.fit(X_train, y_train)
    # SOLUTION END

    # Make predictions using the testing set
    # SOLUTION START
    predictions = model.predict(X_test)
    # SOLUTION END

    # Fill in the blanks
    # SOLUTION START
    coefficients = model.coef_  # The coefficients i.e. the slope
    intercept = model.intercept_  # The intercept
    mse = mean_squared_error(y_test, predictions)  # Mean Squared Error
    rmse = math.sqrt(mse)  # Root Mean Squared Error
    r2score = r2_score(y_test, predictions)  # R-squared score
    # SOLUTION END

    return (coefficients, intercept, mse, rmse, r2score)