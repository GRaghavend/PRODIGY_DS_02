import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns', 15)

#Cleansing and importing the pandas dataframe
df= pd.read_csv('/Users/raghavender/Datasets:Research/titanic/test.csv')
df= df[["PassengerId","Pclass","Name","Age","Sex","Ticket","Fare"]].copy()
df= df.rename(columns={'Ticket':'TicketName'})

#Handling missing values and null values

Null_value = df.isna().sum()
Duplicate_values=df.duplicated(subset=('PassengerId','Pclass','Name','Age','Sex','TicketName','Fare')).sum()
print("The number of null values is: \n",Null_value)
print("\nThe number of duplicate values is:",Duplicate_values)

#Feature Understanding (Univerate analysis)

ax1 = df['Age'].value_counts().plot(kind='bar',title='Top Ages of Titanic Dataset')
ax2 = df['Sex'].value_counts().plot(kind='bar',title='Male Female Ratio')
plt.show()


#Feature Relationship

df.plot(kind='scatter',x='Age',y='Fare',title='Scatter plot between Age and the fare')
plt.xlabel("Age groups")
plt.ylabel("Fare in dollars")
plt.ylim(0,10)
plt.show()

#To find the correlation between the coloumns
df_corr=df[['Age','Fare','Pclass']].dropna().corr()
sns.heatmap(df_corr)
plt.show()