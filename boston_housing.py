# Importiamo le librerie base.
import pandas as pd 
import numpy as np

# Importiamo il dataset.
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Creiamo in DataFrame e mostriamo le prime 5 rilevazioni.
boston = pd.read_csv(url, sep='\s+', names=features)
boston.head()

# Features
# 1. CRIM      per capita crime rate by town
# 2. ZN        proportion of residential land zoned for lots over 
#              25,000 sq.ft.
# 3. INDUS     proportion of non-retail business acres per town
# 4. CHAS      Charles River dummy variable (= 1 if tract bounds 
#              river; 0 otherwise)
# 5. NOX       nitric oxides concentration (parts per 10 million)
# 6. RM        average number of rooms per dwelling
# 7. AGE       proportion of owner-occupied units built prior to 1940
# 8. DIS       weighted distances to five Boston employment centres
# 9. RAD       index of accessibility to radial highways
# 10. TAX      full-value property-tax rate per $10,000
# 11. PTRATIO  pupil-teacher ratio by town
# 12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
#              by town
# 13. LSTAT    % lower status of the population
# 14. MEDV     Median value of owner-occupied homes in $1000's

# Creiamo due numpy array con i dati che interessano alla nostra analisi.
X = boston['RM'].values
y = boston['MEDV'].values

# Dividiamo i nostri dati per creare due distinti set: uno per l'addestramento del nostro modello ed uno per testarlo.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Creiamo il nostro modello con Keras.
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()                                              # Creiamo il modello
layer = Dense(1, input_dim = 1)                                   # Creiamo un unico strato con 1 input ed 1 output 
model.add(layer)                                                  # Aggiungiamo lo strato al modello
model.compile( optimizer = 'sgd', loss = 'mean_squared_error' )   # Assegniamo ottimizzatore e funzione di costo          

# Ora che abbiamo impostato il modello procediamo con il suo addestramento
model.fit(X_train, y_train, epochs = 100)   

# Ora che abbiamo impostato il modello procediamo con il suo addestramento
model.fit(X_train, y_train, epochs = 100)   

# Andiamo a conoscere i pesi della nostra retta di regressione 
pesi = model.get_weights()
print('Inclinazione: ',pesi[0])
print('Intercetta: ',pesi[1])

 # Calcoliamo una serie di previsioni
 y_pred = model.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Numero medio di stanze per edificio')
plt.ylabel('Valore medio in migliaia di dollari')