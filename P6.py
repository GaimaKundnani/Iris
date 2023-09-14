import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Load Dataset
iris=load_iris()
print(dir(iris))
#Split Data
feat=iris.data
print("Data Shape:",feat.shape)
print(feat.shape)
tar=iris.target
print("Target Shape:",tar.shape)
X_train,X_test,y_train,y_test=train_test_split(feat,tar,test_size=0.3,random_state=3)
model=LogisticRegression()
model.fit(X_train,y_train)
pre=model.predict(X_test)
print("Accuaracy:",model.score(X_test,y_test)*100)
new_samples = np.array([[5.1, 3.5, 1.4, 0.2],  
                         [6.5, 3.0, 5.5, 1.8],
                         [4.9, 3.0, 1.4, 0.2]])    
new_predictions = model.predict(new_samples)
species_names = {0: 'Setosa', 1: 'Versicolour', 2: 'Virginica'}
predicted_species = [species_names[pred] for pred in new_predictions]
for i, sample in enumerate(new_samples):
    print(f"Sample {i + 1}: {sample} => Predicted Species: {predicted_species[i]}")
