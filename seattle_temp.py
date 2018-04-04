import numpy as np
import pandas as pd
from datetime import date as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

# rain= pd.read_csv('seattleweather_1948-2017.csv', parse_dates=True, index_col='DATE')
rain= pd.read_csv('seattleweather_1948-2017.csv')
print(rain.head())
print(rain.tail())
print(rain.shape)
print(rain.info())

rain= rain.fillna(0)
print(rain.info())

#Plots of all the data
plt.plot(rain['TMIN'], rain['TMAX'], marker='.', linestyle='none', color='red')
plt.xlabel('TMIN')
plt.ylabel('TMAX')
plt.title('All data TMin vs TMax')
plt.show()

# plt.hist(rain['TMAX'], bins=100)
# plt.ylabel('Max Temp')
# plt.show()
#
# plt.hist(rain['TMIN'], bins=50)
# plt.ylabel('Min Temp')
# plt.show()

X= rain.drop(['DATE','TMAX', 'PRCP', 'RAIN'], axis=1).values
y= rain['TMAX'].values
y= y.reshape(-1, 1)

#train/test split for LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg= linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print('R2_test: {}'. format(reg.score(X_test, y_test)))
rsme= np.sqrt(mean_squared_error(y_test, y_pred))
print('mse_ytest_vs_ypred', rsme)

#plot data and regression line
reg.fit(X, y)
prediction = np.linspace(min(X), max(X)).reshape(-1, 1)
# print(prediction)
print('R2_data: {}'.format(reg.score(X, y)))
plt.scatter(X, y, marker='.', color='red')
plt.plot(prediction, reg.predict(prediction), color='black', linewidth=1)
plt.ylim([0, 105])
plt.xlim([0, 75])
plt.ylabel('TMAX')
plt.xlabel('TMIN')
plt.title('TMIN vs TMAX')
plt.show()


#See if model by month is more accurate
split_date = rain['DATE'].str.split('/').tolist()
df=  pd.DataFrame(split_date)
df.columns=['month', 'day', 'year']
df['month']= df['month'].astype(int)
df['day']= df['day'].astype(int)
df['year']= df['year'].astype(int)
# print(df.head())
# print(df.shape)
# print(df.info())
df_list= [df, rain]
rain_split = pd.concat([df, rain], axis= 1)
# rain_split= rain_split.set_index('year')
# print(rain_split.head())
rain_split= rain_split.reset_index()
rain_split= rain_split.set_index('month')

#Scatter plots by month
def monthly_temp():
    i=1
    for item in rain_split :
        plt.plot(rain_split.loc[i]['TMIN'], rain_split.loc[i]['TMAX'], marker='.', linestyle='none', color='red')
        plt.axis([0, 75, 0, 105])
        plt.xlabel('TMIN')
        plt.ylabel('TMAX')
        plt.title(f'Month {i}')
        i=i + 1
        plt.show()
    i=8
    for item in rain_split :
        if i <= 12 :
            plt.plot(rain_split.loc[i]['TMIN'], rain_split.loc[i]['TMAX'], marker='.', linestyle='none', color='red')
            plt.ylim([0, 105])
            plt.xlim([0, 75])
            plt.xlabel('TMIN')
            plt.ylabel( 'TMAX')
            plt.title(f'Month {i}')
            i=i + 1
            plt.show()

monthly_temp()
#prepare data for model analysis
#Crossvalidation and model performance test
# reg= linear_model.LinearRegression()
cv_results= cross_val_score(reg, X, y, cv=5)
print(cv_results)
print(np.mean(cv_results))

#See if model by month is more accurate
split_date = rain['DATE'].str.split('/').tolist()
df=  pd.DataFrame(split_date)
df.columns=['month', 'day', 'year']
df['month']= df['month'].astype(int)
df['day']= df['day'].astype(int)
df['year']= df['year'].astype(int)
# print(df.head())
# print(df.shape)
# print(df.info())
df_list= [df, rain]
rain_split = pd.concat([df, rain], axis= 1)
# rain_split= rain_split.set_index('year')
# print(rain_split.head())
rain_split= rain_split.reset_index()
rain_split= rain_split.set_index('month')

# #create monthyly dataframes for linear_model(THIS LOOP WORKS TO PRINT THE OUTPUT BUT NOT TO USE THE DataFrame AS AN INPUT)
# i = 1
# for line in rain_split:
#     if i <= 12 :
#         temp_i = rain_split.loc[i]
#         del temp_i['day']
#         del temp_i['year']
#         del temp_i['DATE']
#         del temp_i['PRCP']
#         del temp_i['RAIN']
#         i = i + 1
# i = 8
# for line in rain_split:
#     if i <= 12 :
#         temp_i = rain_split.loc[i]
#         del temp_i['day']
#         del temp_i['year']
#         del temp_i['DATE']
#         del temp_i['PRCP']
#         del temp_i['RAIN']
#         print(temp_i.head())
#         print(temp_i.tail())
#         print(temp_i.info())
#         i = i + 1

#CREATE DATAFRAME WITH JANARY TEMPERATURE DATA
del rain_split['day']
del rain_split['year']
del rain_split['DATE']
del rain_split['index']
del rain_split['RAIN']
print(rain_split.head())
print(rain_split.tail())
print(rain_split.info())

rain_split_slope, rain_split_intercept= np.polyfit(rain_split['TMIN'], rain_split['TMAX'], 1)
# print(rain_split_slope, rain_split_intercept)

#JANUARY DATA
X= rain_split.loc[1].drop(['TMAX', 'PRCP'], axis=1).values
y= rain_split.loc[1]['TMAX'].values
y= y.reshape(-1, 1)

#train/test split for LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg= linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print('R2_test: {}'. format(reg.score(X_test, y_test)))
rsme= np.sqrt(mean_squared_error(y_test, y_pred))
print('mse_ytest_vs_ypred', rsme)

#plot data and regression line
reg.fit(X, y)
prediction = np.linspace(min(X), max(X)).reshape(-1, 1)
# print(prediction)
print('R2_data: {}'.format(reg.score(X, y)))
plt.scatter(X, y, color='red', marker='.')
plt.plot(prediction, reg.predict(prediction), color='black', linewidth=1)
plt.ylim([0, 105])
plt.xlim([0, 75])
plt.ylabel('TMAX')
plt.xlabel('TMIN')
plt.title('JANARY TMin vs TMax')
plt.show()

#Crossvalidation and model performance test
# reg= linear_model.LinearRegression()
cv_results= cross_val_score(reg, X, y, cv=10)
print(cv_results)
print(np.mean(cv_results))

X= rain_split.loc[11].drop(['TMAX', 'PRCP'], axis=1).values
y= rain_split.loc[11]['TMAX'].values
y= y.reshape(-1, 1)

#train/test split for LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg= linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print('R2_test: {}'. format(reg.score(X_test, y_test)))
rsme= np.sqrt(mean_squared_error(y_test, y_pred))
print('mse_ytest_vs_ypred', rsme)

#plot data and regression line
reg.fit(X, y)
prediction = np.linspace(min(X), max(X)).reshape(-1, 1)
# print(prediction)
print('R2_data: {}'.format(reg.score(X, y)))
plt.scatter(X, y, marker='.', color='red')
plt.plot(prediction, reg.predict(prediction), color='black', linewidth=1)
plt.ylim([0, 105])
plt.xlim([0, 75])
plt.ylabel('TMAX')
plt.xlabel('TMIN')
plt.title('November TMin vs TMax')
plt.show()

#Crossvalidation and model performance test
# reg= linear_model.LinearRegression()
cv_results= cross_val_score(reg, X, y, cv=10)
print(cv_results)
print(np.mean(cv_results))

X= rain_split.loc[12].drop(['TMAX', 'PRCP'], axis=1).values
y= rain_split.loc[12]['TMAX'].values
y= y.reshape(-1, 1)

#train/test split for LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg= linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print('R2_test: {}'. format(reg.score(X_test, y_test)))
rsme= np.sqrt(mean_squared_error(y_test, y_pred))
print('mse_ytest_vs_ypred', rsme)

#plot data and regression line
reg.fit(X, y)
prediction = np.linspace(min(X), max(X)).reshape(-1, 1)
# print(prediction)
print('R2_data: {}'.format(reg.score(X, y)))
plt.scatter(X, y, marker='.', color='red')
plt.plot(prediction, reg.predict(prediction), color='black', linewidth=1)
plt.ylim([0, 105])
plt.xlim([0, 75])
plt.ylabel('TMAX')
plt.xlabel('TMIN')
plt.title('December TMin vs TMax')
plt.show()

#Crossvalidation and model performance test
# reg= linear_model.LinearRegression()
cv_results= cross_val_score(reg, X, y, cv=10)
print(cv_results)
print(np.mean(cv_results))
