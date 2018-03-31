import numpy as np
import pandas as pd
from datetime import date as dt

import matplotlib.pyplot as plt

# rain= pd.read_csv('seattleweather_1948-2017.csv', parse_dates=True, index_col='DATE')
rain= pd.read_csv('seattleweather_1948-2017.csv')

# vals <- as.character(60:70)
# as.POSIXct(paste0("19",vals), format = "%Y")

# print(rain.head())
# print(rain.tail())
# print(rain.shape)
# print(rain.info())

split_date = rain['DATE'].str.split('/').tolist()
df=  pd.DataFrame(split_date)
df.columns=['month', 'day', 'year']

# df['viz'] = (df['viz'] !='n').astype(int)

df['month']= df['month'].astype(int)
df['day']= df['day'].astype(int)
df['year']= df['year'].astype(int)
# print(df.head())
# print(df.shape)
# print(df.info())

df_list= [df, rain]
rain_split = pd.concat([df, rain], axis= 1)
# print(rain_split.head())

rain_split= rain_split.set_index(['month', 'day'])
rain_split= rain_split.sort_index()
# print(rain_split[1400:1450])
# print(rain_split.tail(20))
# print(rain_split.shape)

rain_months= rain_split['RAIN'].groupby('month').sum()
count_months= rain_split['RAIN'].groupby('month').count()
# print(rain_months)
# print(count_months)
prob_rain_month= rain_months/count_months
prob_rain= pd.DataFrame([prob_rain_month])
prob_rain_v= prob_rain.transpose()
prob_rain_v['month']= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
prob_rain_v= prob_rain_v.set_index('month')
# print(prob_rain_v)
# print(prob_rain_v.info())

#will it rain on September
m= input("Choose a month, i.e Jan, Feb ...  ")
d= input("Choose a day  ")
d= int(d)
mon_31= ['Jan', 'Mar', 'May', 'Jul', 'Aug', 'Dec' ]
mon_30= ['Apr', 'Jun', 'Sep', 'Nov']
if m in mon_31:
    d= 31
elif m in mon_30:
    d= 30
else:
    d= 28
#print(d)
p=prob_rain_v.loc[m]

yes_no=np.random.binomial(1, p, size=31)[d]

if yes_no == 0 :
    print('\n' * 2)
    print('*******************************')
    print(f'It will not rain in Seattle on {m} {d}')
    print('*******************************')
    print('\n' * 2)
else:
    print('\n' * 2)
    print('*******************************')
    print(f'It will rain in Seattle on {m} {d}')
    print('*******************************')
    print('\n' * 2)






# m= input("Choose a month ")
# d= input("Choose a day ")
# d= int(d)
# prob= prob_month[m][d]
# if prob <= 0.5 :
#     print('It will not rain in Seattle today')
# else :
#     print('It will rain in Seattle today')
# print(prob)








# #this works for 1 month at a time
# jan_prob= []
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(1, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(1, [n]), ].count()
#         jan_prob.append(rain_days/count_days)
#         n = n + 1
# jan_prob_s= pd.Series(jan_prob)
#
# # print('jan_prob')
# # # print(jan_prob)
# # print(jan_prob_s)

# feb_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(2, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(2, [n]), ].count()
#         feb_prob.append(rain_days/count_days)
#         n = n + 1
# feb_prob_s= pd.Series(feb_prob)
# # print('feb_prob')
# # # print(feb_prob)
# # print(feb_prob_s)
#
# mar_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(3, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(3, [n]), ].count()
#         mar_prob.append(rain_days/count_days)
#         n = n + 1
# mar_prob_s= pd.Series(mar_prob)
# # print('mar_prob')
# # print(mar_prob_s)
#
# apr_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(4, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(4, [n]), ].count()
#         apr_prob.append(rain_days/count_days)
#         n = n + 1
# apr_prob_s= pd.Series(apr_prob)
# # print('apr_prob')
# # print(apr_prob_s)
#
# may_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(5, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(5, [n]), ].count()
#         may_prob.append(rain_days/count_days)
#         n = n + 1
# may_prob_s= pd.Series(may_prob)
# # print('may_prob')
# # print(may_prob_s)
#
# jun_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(6, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(6, [n]), ].count()
#         jun_prob.append(rain_days/count_days)
#         n = n + 1
# jun_prob_s= pd.Series(jun_prob)
# # print('jun_prob')
# # print(jun_prob_s)
#
# jul_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(7, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(7, [n]), ].count()
#         jul_prob.append(rain_days/count_days)
#         n = n + 1
# jul_prob_s= pd.Series(jul_prob)
# # print('jul_prob')
# # print(jul_prob_s)
# #
# aug_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(8, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(8, [n]), ].count()
#         aug_prob.append(rain_days/count_days)
#         n = n + 1
# aug_prob_s= pd.Series(aug_prob)
# # print('aug_prob')
# # print(aug_prob_s)
# #
# sep_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(9, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(9, [n]), ].count()
#         sep_prob.append(rain_days/count_days)
#         n = n + 1
# sep_prob_s= pd.Series(sep_prob)
# # print('sep_prob')
# # print(sep_prob_s)
# #
# oct_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(10, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(10, [n]), ].count()
#         oct_prob.append(rain_days/count_days)
#         n = n + 1
# oct_prob_s= pd.Series(oct_prob)
# # print('oct_prob')
# # print(oct_prob_s)
#
# nov_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(11, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(11, [n]), ].count()
#         nov_prob.append(rain_days/count_days)
#         n = n + 1
# nov_prob_s= pd.Series(nov_prob)
# # print('nov_prob')
# # print(nov_prob_s)
#
# dec_prob=[]
# n=1
# for entry in rain_split['RAIN'] :
#     if n <= 31 :
#         rain_days= rain_split['RAIN'].loc[(12, [n]), ].sum()
#         count_days= rain_split['RAIN'].loc[(12, [n]), ].count()
#         dec_prob.append(rain_days/count_days)
#         n = n + 1
# dec_prob_s= pd.Series(dec_prob)
# # print('dec_prob')
# # print(dec_prob_s)
#
# prob_mon= pd.DataFrame([jan_prob_s, feb_prob_s, mar_prob_s, apr_prob_s, may_prob_s, jun_prob_s, jul_prob_s, aug_prob_s, sep_prob_s, oct_prob_s, nov_prob_s, dec_prob_s])
# prob_mon= prob_mon.transpose()
# prob_mon.columns= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# prob_mon['Day']= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# prob_month= prob_mon.set_index('Day')
# print(prob_month)
#
# m= input("Choose a month ")
# d= input("Choose a day ")
# d= int(d)
# prob= prob_month[m][d]
# if prob <= 0.5 :
#     print('It will not rain in Seattle today')
# else :
#     print('It will rain in Seattle today')
# print(prob)

#




# names2 = names.copy()
# total_births_by_year = names2.groupby('Year')['Count'].transform('sum')
# names2['pct_name']= (names2['Count']/total_births_by_year)* 100
# print('NAMES DATAFRAME WITH PCT NAME ADDED')
# print(names2.tail())
# print(names2.shape)
