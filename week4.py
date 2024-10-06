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
	df = pd.read_csv('data/realestate.csv')
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
    global df_X, df_y
    
    # Prepare features and target variable
    X = df_X
    y = df_y.values.ravel()  # Flatten to 1D array
    
    # Split the data into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reset indices of the resulting DataFrames
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = pd.DataFrame(y_train, columns=['Y house price of unit area']).reset_index(drop=True)  # Convert to DataFrame
    y_test = pd.DataFrame(y_test, columns=['Y house price of unit area']).reset_index(drop=True)    # Convert to DataFrame
    
    return X_train, X_test, y_train, y_test
    # SOLUTION END

X_train, X_test, y_train, y_test = split()


def makeModel():
    '''
    This function will create a linear regression object, train the model using the training sets,
    make predictions using the testing set, and return coefficients, intercept, mse, rmse, r2score.
    '''
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coefficients = model.coef_
    intercept = model.intercept_
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return coefficients, intercept, mse, rmse, r2
  
coefficients1, intercept1, mse1, rmse1, r2_score1 = makeModel()