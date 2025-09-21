import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv(r"C:\Users\amar_shukla\Desktop\Rain\jan.csv").head(31)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['date_num'] = df['date'].map(lambda x: x.toordinal())
v = input("Enter your date in format d-m-y : ")
s = pd.to_datetime(v, format='%d-%m-%Y')
s_num = s.toordinal()

#print(df_2023.to_string())
reg=linear_model.LinearRegression()
reg.fit(df[['date_num']],df['P_rain'])
value = reg.predict(pd.DataFrame([[s_num]], columns=['date_num']))
print(f"Predicted rainfall on {v}: {value[0]:.2f}")


plt.scatter(df['date_num'], df['P_rain'], color='blue', label='Actual')
plt.plot(df['date_num'], reg.predict(df[['date_num']]), color='red', label='Regression Line')
plt.scatter(s_num, value, color='green', label='Prediction')
plt.legend()
plt.xlabel('Date (ordinal)')
plt.ylabel('Rainfall')
plt.title('Rainfall Prediction')
plt.show()