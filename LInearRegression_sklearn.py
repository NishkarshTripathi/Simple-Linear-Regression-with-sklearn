import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import seaborn as sb
import matplotlib.pyplot as plt
boston = load_boston()

#df = pd.DataFrame(boston)
#print(df_x.head())
#c = df.corr()

##Check the correlation using heatmap
#sb.heatmap(c, vmax=1, vmin=0, square=True)
#plt.show

df_x = pd.DataFrame(boston.data, columns=boston.feature_names)
df_y = pd.DataFrame(boston.target)

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

reg.fit(x_train, y_train)

print(reg.score(x_test))
