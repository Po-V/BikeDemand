import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
 
bike_data = pd.read_csv('hour.csv')
#bike_data.info()

bike_data = bike_data.drop(['instant', 'dteday', 'casual', 'registered'], axis = 1)

#print(bike_data.isna().sum())

#visualise continuous data
'''
plt.subplot(2,2, 1)
plt.title('Temperature vs demand')
plt.scatter(bike_data['temp'], bike_data['cnt'], s=2)

plt.subplot(2,2,2)
plt.title('aTemperature vs demand')
plt.scatter(bike_data['atemp'], bike_data['cnt'], s=2)

plt.subplot(2,2,3)
plt.title('hum vs demand')
plt.scatter(bike_data['hum'], bike_data['cnt'], s=2)

plt.subplot(2,2,4)
plt.title('windspeed vs demand')
plt.scatter(bike_data['windspeed'], bike_data['cnt'], s=2)

plt.tight_layout()
plt.show()

#demand is not normally distributed
#temp and atemp shows similar spread of data (possible high corelation)
#windspeed remain constant until a point & goes down in demand after
#little change in demand with change in humidity
'''

#visualize categorical variables
'''
plt.subplot(3,3,1)
plt.title('Average demand per season')
seasons = bike_data['season'].unique()
avg_cnt = bike_data.groupby('season').mean()['cnt']
plt.bar(seasons, avg_cnt)

plt.subplot(3,3,2)
plt.title('Average demand per year')
year = bike_data['yr'].unique()
avg_cnt = bike_data.groupby('yr').mean()['cnt']
plt.bar(year, avg_cnt)

plt.subplot(3,3,3)
plt.title('Average demand per month')
month = bike_data['mnth'].unique()
avg_cnt = bike_data.groupby('mnth').mean()['cnt']
plt.bar(month, avg_cnt)

plt.subplot(3,3,4)
plt.title('Average demand per hr')
hour = bike_data['hr'].unique()
avg_cnt = bike_data.groupby('hr').mean()['cnt']
plt.bar(hour, avg_cnt)

plt.subplot(3,3,5)
plt.title('Average demand per holiday')
holiday = bike_data['holiday'].unique()
avg_cnt = bike_data.groupby('holiday').mean()['cnt']
plt.bar(holiday, avg_cnt)

plt.subplot(3,3, 6)
plt.title('Average demand per weekday')
weekday = bike_data['weekday'].unique()
avg_cnt = bike_data.groupby('weekday').mean()['cnt']
plt.bar(weekday, avg_cnt)

plt.subplot(3,3, 7)
plt.title('Average demand per workingday')
workingday = bike_data['workingday'].unique()
avg_cnt = bike_data.groupby('workingday').mean()['cnt']
plt.bar(workingday, avg_cnt)

plt.subplot(3,3,8)
plt.title('Average demand per weathersit')
weathersit = bike_data['weathersit'].unique()
avg_cnt = bike_data.groupby('weathersit').mean()['cnt']
plt.bar(weathersit, avg_cnt)

plt.tight_layout()
plt.show()

#Workingday does not have much affect on demand, can be removed
#year can be removed because there is only 2 yrs- does not provide much input
#weekday can be removed because it does not have much effect
'''

#check for outliers
'''
print(bike_data['cnt'].describe())
print(bike_data['cnt'].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99]))
#50% of the data is between 40 and 281 which is very far from min and max
#5% of the time demand is 5 bikes, only 1% of the time, demand is above 782 (may be considered as outliers)
'''

#testing linearity using correlation coefficient matrix
correlation = bike_data[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
#atemp has high correlation with temp so it can be dropped (multicollinearity)
#humidity & windspeed has some correlation; however windspeed and demand has low correlation so can be dropped.

bike_data = bike_data.drop(['workingday', 'yr', 'weekday', 'windspeed', 'atemp'], axis = 1)

#check the autocorrelation in demand 
#There is high autocorrelation in demand 
df_auto = pd.to_numeric(bike_data['cnt'], downcast = 'float')
plt.acorr(df_auto, maxlags=12)

#solving problem of normality by taking log transfrom of original log normal distribution
'''
demand = bike_data['cnt']
demand_log = np.log(demand)
plt.figure()
demand.hist()
plt.figure()
demand_log.hist()
'''
bike_data['cnt'] = np.log(bike_data['cnt'])

#fix autocorrelation in demand by creating timelag features

t_1 = bike_data['cnt'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bike_data['cnt'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bike_data['cnt'].shift(+3).to_frame()
t_3.columns = ['t-3']

bike_data = pd.concat([bike_data, t_1, t_2, t_3], axis = 1)
bike_data = bike_data.dropna()

#create dummy variables and drop first to avoid dummy variables trap
print(bike_data.dtypes)

bike_data['season'] = bike_data['season'].astype('category')
bike_data['holiday'] = bike_data['holiday'].astype('category')
bike_data['weathersit'] = bike_data['weathersit'].astype('category')
bike_data['mnth'] = bike_data['mnth'].astype('category')
bike_data['hr'] = bike_data['hr'].astype('category')

bike_data = pd.get_dummies(bike_data, drop_first = True)

#demand is time series data. So need to manually split it.
y = bike_data[['cnt']]
X = bike_data.drop(['cnt'], axis = 1)

X_train = X.values[0 : int(0.7 *len(X))]
X_test = X.values[int(0.7* len(X)) : len(X)]

y_train = y.values[0 : int(0.7 *len(X))]
y_test = y.values[int(0.7* len(X)) : len(X)]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

r2_train = regressor.score(X_train, y_train)
r2_test = regressor.score(X_test, y_test)

y_predict = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))

#to calculate RMSLE, convert y values back using exp

y_test_e = []
y_predict_e = []

for i in range(0, len(y_test)):
    y_test_e.append(np.exp(y_test[i]))
    y_predict_e.append(np.exp(y_predict[i]))
    
log_sq_sum = 0.0

for i in range(0, len(y_test)):
    log_a = np.log(y_test_e[i] + 1)
    log_p = np.log(y_predict_e[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = np.sqrt(log_sq_sum/len(y_test))


















































