
#%%
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import datasets
boston_data = datasets.load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
df_boston['PRICE'] = pd.Series(boston_data.target)
df_boston.head()

#Hello
##Hellow Markdown
sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.hist(df_boston['PRICE'],color ="brown", bins=30)
plt.xlabel("House prices in $1000")
plt.show()


correlation_matrix=df_boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = df_boston['PRICE']

#Hello!!!!
for i, col in enumerate(features):
   plt.subplot(1, len(features) , i+1)
   x = df_boston[col]
   y = target
   plt.scatter(x, y,color='green', marker='o')
   plt.title("Variation in House prices")
   plt.xlabel(col)
   plt.ylabel('"House prices in $1000"')

plt.show()
print("All Done!")



# %%
