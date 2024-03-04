import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
df_car = pd.read_excel("/content/EDA Cars.xlsx")
df_car.head(30)
df_car.shape
df_car.info()
df_car.describe()
df_car.info()
df_car.describe(include=['object', 'int','float'])
df_car.INCOME
df_car[df_car['INCOME'] == '@@']
df_car['INCOME'] = df_car['INCOME'].replace(to_replace = '@@', value= np.nan)
df_car['INCOME'] = df_car['INCOME'].astype(float)
df_car[df_car['INCOME'] == '@@']
df_car.isnull().sum()
median_income = df_car['INCOME'].median()
median_travel_time = df_car['TRAVEL TIME'].median()
median_miles_clocked = df_car['MILES CLOCKED'].median()
median_car_age = df_car['CAR AGE'].median()
median_postal_code = df_car['POSTAL CODE'].median()
df_car['INCOME'].replace(np.nan, median_income, inplace = True)
df_car['TRAVEL TIME'].replace(np.nan, median_travel_time, inplace = True)
df_car['MILES CLOCKED'].replace(np.nan, median_miles_clocked, inplace = True)
df_car['CAR AGE'].replace(np.nan, median_car_age, inplace = True)
df_car['POSTAL CODE'].replace(np.nan, median_postal_code, inplace = True)
mode_sex = df_car['SEX'].mode().values[0]
mode_martial_status = df_car['MARITAL STATUS'].mode().values[0]
mode_education = df_car['EDUCATION'].mode().values[0]
mode_job = df_car['JOB'].mode().values[0]
mode_use = df_car['USE'].mode().values[0]
mode_city = df_car['CITY'].mode().values[0]
mode_car_type = df_car['CAR TYPE'].mode().values[0]
df_car['SEX']= df_car['SEX'].replace(np.nan, mode_sex)
df_car['MARITAL STATUS']= df_car['MARITAL STATUS'].replace(np.nan, mode_martial_status)
df_car['EDUCATION']= df_car['EDUCATION'].replace(np.nan, mode_education)
df_car['JOB']= df_car['JOB'].replace(np.nan, mode_job)
df_car['USE']= df_car['USE'].replace(np.nan, mode_use)
df_car['CITY']= df_car['CITY'].replace(np.nan, mode_city)
df_car['CAR TYPE']= df_car['CAR TYPE'].replace(np.nan, mode_car_type)
df_car.isnull().sum()
duplicate = df_car.duplicated()
print(duplicate.sum())
df_car[duplicate]
df_car.drop_duplicates(inplace=True)
dup = df_car.duplicated()
dup.sum()
df_car.boxplot(column=['INCOME'])
plt.show()
def remove_outlier(col):
  sorted(col)
  Q1, Q3 = col.quantile([0.25, 0.75])
  IQR = Q3-Q1
  lower_range = Q1-(1.5 * IQR)
  upper_range = Q3+(1.5 * IQR)
  return lower_range, upper_range
lower_income, upper_income = remove_outlier(df_car['INCOME'])
df_car['INCOME'] = np.where(df_car['INCOME'] > upper_income, upper_income, df_car['INCOME'])
df_car['INCOME'] = np.where(df_car['INCOME'] < lower_income, lower_income, df_car['INCOME'])
df_car.boxplot(column=['INCOME'])
plt.show()
df_car.corr()
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale
df_car['INCOME'] = std_scale.fit_transform(df_car[['INCOME']])
df_car['TRAVEL TIME'] = std_scale.fit_transform(df_car[['TRAVEL TIME']])
df_car['CAR AGE'] = std_scale.fit_transform(df_car[['CAR AGE']])
df_car['POSTAL CODE'] = std_scale.fit_transform(df_car[['POSTAL CODE']])
df_car['MILES CLOCKED'] = std_scale.fit_transform(df_car[['MILES CLOCKED']])
df_car.head()
dummies = pd.get_dummies(df_car[['MARITAL STATUS', 'SEX', 'EDUCATION', 'JOB', 'USE', 'CAR TYPE', 'CITY']],                          columns = ['MARITAL STATUS', 'SEX', 'EDUCATION', 'JOB', 'USE', 'CAR TYPE', 'CITY'], prefix = ['married', 'sex', 'education', 'job', 'use', 'cartype', 'ciyt'], drop_first= True).head()
dummies.head()
columns = ["MARITAL STATUS", "SEX", "EDUCATION","JOB","USE", "CAR TYPE", "CITY"]
df_car = pd.concat([df_car, dummies], axis=1)
df_car.drop(columns, axis= 1, inplace=True )
df_car.head()
