import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

dataframe = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

X = dataset[:,0:13]
Y = dataset[:,13]
sc = StandardScaler()
X = sc.fit_transform(X)

scy = StandardScaler()
Y = scy.fit_transform(Y.reshape(-1, 1))
# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(output_dim = 7, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(output_dim = 7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

regressor = KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5)
parameters = {'batch_size': [5, 32, 64], 'epochs': [100, 500]}
grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=10)
grid_search = grid_search.fit(X, Y)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print best_parameters, best_accuracy