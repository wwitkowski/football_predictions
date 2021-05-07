import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam


class FootballPoissonModel():

	
	def __init__(self, data):
		self.data = pd.concat([data[['team1', 'team2', 'score1']].assign(home=1).rename(
			columns={'team1': 'team', 'team2': 'opponent', 'score1': 'goals'}),
			data[['team2', 'team1', 'score2']].assign(home=0).rename(
			columns={'team2': 'team', 'team1': 'opponent', 'score2': 'goals'})])


	def fit(self):
		self.model = smf.glm(formula="goals ~ home + team + opponent", data=self.data, 
			family=sm.families.Poisson()).fit()

	@property
	def summary(self):
		return self.model.summary()


	def predict_goals(self, data):
		home_goals_pred = self.model.predict(data.assign(home=1).rename(
						columns={'team1': 'team', 'team2': 'opponent'}))
		away_goals_pred = self.model.predict(data.assign(home=0).rename(
						columns={'team2': 'team', 'team1': 'opponent'}))

		return home_goals_pred, away_goals_pred


	def predict_chances(self, data, max_goals=10):

		htg, atg = self.predict_goals(data)
		team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [htg, atg]]


		match_pred = [[np.outer(np.array([i[j] for i in team_pred[0]]), np.array([i[j] for i in team_pred[1]]))] for j in range(0, np.shape(team_pred)[2])]


		res = [[np.sum(np.tril(match_pred[i][0], -1)), np.sum(np.diag(match_pred[i][0])), np.sum(np.triu(match_pred[i][0], 1))] for i in range(0, len(match_pred))] 

		return zip(*res)


class NeuralNetworkModel():


	def __init__(self, name=None):
		self.name = name

		if self.name is not None:
			self.model = tf.keras.models.load_model(f'models\\{self.name}.hdf5')


	def build(self, n_features, activations, nodes, bias=None):
		optimizer = 'sgd'
		loss = tf.keras.losses.MeanSquaredError()
		metrics = ['mae']

		if bias is not None:
				bias = tf.keras.initializers.Constant(bias)

		self.model = tf.keras.Sequential()

		# First layer
		self.model.add(tf.keras.layers.Dense(nodes[0], activation=activations[0], input_shape=(n_features,)))

		# Hidden layers
		for i in range(1, len(activations)):
			self.model.add(tf.keras.layers.Dense(nodes[i], activation=activations[i]))
			#self.model.add(tf.keras.layers.Dropout(0.05))

		# Output layer
		self.model.add(tf.keras.layers.Dense(2, bias_initializer=bias))

		self.model.compile(
			optimizer=optimizer,
			loss=loss,
			metrics=metrics)


	def train(self, X_train, y_train, X_val, y_val, verbose=1, batch_size=256, epochs=200):

		# checkpoint = tf.keras.callbacks.ModelCheckpoint(f"models\\nn_model_{date.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.hdf5",
		# 												 monitor='val_loss',
		# 												 save_best_only=True)

		early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0)

		self.history = self.model.fit(X_train, y_train,
									  batch_size=batch_size,
									  verbose=verbose,
									  epochs=epochs,
									  #callbacks=[checkpoint],
									  validation_data=(X_val, y_val))

		return self.history


	def predict(self, X):

		y_pred = self.model.predict(X)

		return y_pred


	def save(self, path):

		self.model.save(path)