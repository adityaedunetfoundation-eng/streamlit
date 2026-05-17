import pickle
import pickle 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
iris= load_iris()
x, y = iris.data, iris.target
x_train,x_test,  y_train, y_test = train_test_split(x,y, test_size=0.2)
model  = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

with  open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)



 
print("Model trained and saved as iris_model.pkl")