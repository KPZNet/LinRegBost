
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets


boston = datasets.load_boston()

bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos['PRICE'] = boston.target
bos.head()

sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.hist(bos['PRICE'],color ="brown", bins=30)
plt.xlabel("House prices in $1000")
plt.show()

#Created a dataframe without the price col, since we need to see the
#correlation between the variables
bos_1=pd.DataFrame(boston.data, columns=boston.feature_names)

correlation_matrix=bos_1.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = bos['PRICE']

for i, col in enumerate(features):
   plt.subplot(1, len(features) , i+1)
   x = bos[col]
   y = target
   plt.scatter(x, y,color='green', marker='o')
   plt.title("Variation in House prices")
   plt.xlabel(col)
   plt.ylabel('"House prices in $1000"')

plt.show()
print("All Done!")
