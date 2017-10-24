import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("train.csv").as_matrix()
print("data",data)

clf=DecisionTreeClassifier()

x_train=data[0:21000,1:]
y_train=data[0:21000,0]
x_test=data[21000:,1:]
y_test=data[21000:,0]

#for plotting the pic
disp=x_test[8]
disp.shape=(28,28)
plt.imshow(255-disp,cmap="gray")
plt.show()

p=clf.predict([x_test[0]])
print("prediction",p)
