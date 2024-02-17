import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/user/OneDrive/Desktop/wine.csv")

X_train, X_test, Y_train, Y_test  = train_test_split(df[['density','sulfates','residual_sugar']], df['high_quality'], test_size=0.3)

classifier = KNeighborsClassifier()
classifier.fit(X_train, Y_train)

prediction = classifier.predict(X_test)

correct = np.where(prediction==Y_test, 1,0) .sum()
print(correct)

accuracy = correct/len(Y_test)
print(accuracy)