import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
df = pd.read_csv(r'F:\WORKPROJ\Internship\Student.csv')  ##load data sheet
#print(df)

#print(df.shape)

x=df[['Hours']]
y=df[['Scores']]
reg=linear_model.LinearRegression() ##apply linear regression on the data sheet

reg.fit(x,y)   ##fit the best fit line

#print(reg.score(x,y))


plt.xlabel('Hours')    ##name the x axis
plt.ylabel('Scores')   ##name the y axis
#plt.legend()
plt.scatter(x,y)       #3 can plot scatter form also
plt.plot(x,reg.predict(x))    ## plot the value 
print('the predict score is:')
print(reg.predict([[9.25]]))    ##predict the value

plt.show()    ##show the graph
