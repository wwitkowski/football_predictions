import pandas as pd
import numpy as np

import tensorflow as tf
from data_operations import Bookmaker
from analysis import FeatureTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_absolute_percentage_error, mean_squared_log_error, median_absolute_error
from sklearn.model_selection import KFold

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from models import NeuralNetworkModel, FootballPoissonModel

from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

import matplotlib.pyplot as plt
from kneed import KneeLocator

pd.set_option('mode.chained_assignment', None)


def huber_loss(y_true, y_pred):
	h = tf.keras.losses.Huber()
	return h(y_true, y_pred).numpy()


def logcosh_loss(y_true, y_pred):
	l = tf.keras.losses.LogCosh()	
	return l(y_true, y_pred).numpy()


def poisson_loss(y_true, y_pred):
	p = tf.keras.losses.Poisson()
	return p(y_true, y_pred).numpy()


def evaluate(model, model_name, df_train, target, features, testing=True, df_test=None, cv=None, save=None):

	df_train = df_train[features]
	scaler = StandardScaler()

	if cv:
		kf = KFold(n_splits=cv, shuffle=True, random_state=42)
		df_train_copy = df_train.copy()
		target_copy = target.copy()

		mae = []
		mse = []
		pdev = []
		rmse = []
		mae_test = []
		mse_test = []
		pdev_test = []
		rmse_test = []

		for i, fold in enumerate(kf.split(df_train_copy)):
			print(f'training {model_name}... split: ({i+1}/{cv})')

			train, test = fold			
			# transformer = FeatureTransformer()
			# transformer.fit(df_train_copy.iloc[train])

			# X_train = transformer.transform(df_train_copy.iloc[train])
			# X_train = scaler.fit_transform(X_train.values)
			X_train = scaler.fit_transform(df_train_copy.iloc[train].values)

			# X_val = transformer.transform(df_train_copy.iloc[test])
			# X_val = scaler.transform(X_val.values)
			X_val = scaler.transform(df_train_copy.iloc[test].values)

			y_train = target_copy[train]
			y_val = target_copy[test]

			#y_train = np.sqrt(y_train + 1)

			
			if 'NeuralNet' in model_name:
				# model.build(n_features=X_train.shape[1])
				# model.train(X_train, y_train, X_val, y_val, verbose=0)
				model.fit(X_train, y_train, X_val, y_val, verbose=0)
			elif 'CatBoost' in model_name:
				model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
			elif 'XGB' in model_name:
				model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
			else:
				model.fit(X_train, y_train)

			# y_pred = model.predict(X_val)
			y_pred = model.predict(X_val)
			y_pred = y_pred.clip(min = .01)

			mse.append(mean_squared_error(y_val, y_pred))
			mae.append(mean_absolute_error(y_val, y_pred))
			pdev.append(mean_poisson_deviance(y_val, y_pred))
			rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))

	if testing:
		X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.1, random_state=0)
		# Scale data

		# transformer = FeatureTransformer()
		# transformer.fit(df_train_copy.iloc[train])

		# X_train = transformer.transform(X_train)
		# X_train = scaler.fit_transform(X_train)
		X_train = scaler.fit_transform(X_train.values)

		# X_val = transformer.transform(X_val)
		# X_val = scaler.transform(X_val)
		X_val = scaler.transform(X_val.values)

		#y_train = np.sqrt(y_train + 1)

		if 'NeuralNet' in model_name:
			# model.build(n_features=X_train.shape[1])
			# model.train(X_train, y_train, X_val, y_val, verbose=0)
			model.fit(X_train, y_train, X_val, y_val, verbose=0)
		elif 'CatBoost' in model_name:
			model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
		elif 'XGB' in model_name:
			model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
		else:
			model.fit(X_train, y_train)

		X_test_home = df_test[df_test.home == 1].copy()
		X_test_away = df_test[df_test.home == 0].copy()

		y_test_home = X_test_home.score1.values
		y_test_away = X_test_away.score1.values
		X_test_home = X_test_home[features]
		X_test_away = X_test_away[features]

		# X_test_home = transformer.transform(X_test_home)
		# X_test_away = transformer.transform(X_test_away)

		X_test_home = scaler.transform(X_test_home)
		X_test_away = scaler.transform(X_test_away)

		score1_pred = model.predict(X_test_home)
		score2_pred = model.predict(X_test_away)
		# score1_pred = np.power(model.predict(X_test_home), 2) -1
		# score2_pred = np.power(model.predict(X_test_away), 2) -1
		score1_pred = score1_pred.clip(min = .01)
		score2_pred = score2_pred.clip(min = .01)

		
		if save is not None:
			if model_name in save:
				model.save_model(f'models/{model_name}')

		# mse_test = (mean_squared_error(y_test_home, score1_pred) + mean_squared_error(y_test_away, score2_pred)) /2
		# mae_test = (mean_absolute_error(y_test_home, score1_pred) + mean_absolute_error(y_test_away, score2_pred)) /2
		# pdev_test = (mean_poisson_deviance(y_test_home, score1_pred) + mean_poisson_deviance(y_test_away, score2_pred)) /2
		# rmse_test = (np.sqrt(mean_squared_error(y_test_home, score1_pred)) + np.sqrt(mean_squared_error(y_test_away, score2_pred))) /2
		# mape_test = (mean_absolute_percentage_error(y_test_home, score1_pred) + mean_absolute_percentage_error(y_test_away, score2_pred)) /2
		# msle_test = (mean_squared_log_error(y_test_home, score1_pred) + mean_squared_log_error(y_test_away, score2_pred)) /2
		# medae_test = (median_absolute_error(y_test_home, score1_pred) + median_absolute_error(y_test_away, score2_pred)) /2
		# r2_test = (r2_score(y_test_home, score1_pred) + r2_score(y_test_away, score2_pred)) /2

		# foot_poisson = FootballPoissonModel()
		# home_win, draw, away_win = foot_poisson.predict_chances(score1_pred, score2_pred)
		# over, under = foot_poisson.predict_overs(score1_pred, score2_pred)
		# df_predictions = df_test[df_test.home == 1].copy()

		# predictions = pd.DataFrame(data={'score1_pred': score1_pred.ravel(), 'score2_pred': score2_pred.ravel(),
		# 							 'homewin_pred': np.clip(list(home_win), a_min=0.01, a_max=None), 
		# 							 'draw_pred': np.clip(list(draw), a_min=0.01, a_max=None), 
		# 							 'awaywin_pred': np.clip(list(away_win), a_min=0.01, a_max=None),
		# 							 '>2.5_pred': over, '<2.5_pred': under})

		# df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
		# book = Bookmaker(df_predictions, odds='max', stake=5)
		# book.calculate()

		# #df_predictions.to_csv(f'{model_name}_predictions.csv')

		# returns = df_predictions[['bet_return']].sum().values
		# returns_over = df_predictions[['bet_return_over']].sum().values

	return y_test_home, y_test_away, score1_pred, score2_pred



def optimize(params, name, df_train, df_target):
	

	df_train = df_train[list(params['features'])]
	scaler = StandardScaler()

	kf = KFold(n_splits=4, shuffle=True, random_state=42)
	df_train_copy = df_train.copy()
	target_copy = df_target.copy()

	mae = []
	mse = []
	pdev = []
	rmse = []
	mape = []
	msle = []
	r2 = []
	medae = []
	huber = []
	logcosh = []
	returns = []
	returns_over = []
	poisson = []

	for train, test in kf.split(df_train_copy):
		# transformer = FeatureTransformer()
		# transformer.fit(df_train_copy.iloc[train])

		# X_train = transformer.transform(df_train_copy.iloc[train])
		# X_train = scaler.fit_transform(X_train.values)

		X_train = scaler.fit_transform(df_train_copy.iloc[train].values)

		# X_val = transformer.transform(df_train_copy.iloc[test])
		# X_val = scaler.transform(X_val.values)

		X_val = scaler.transform(df_train_copy.iloc[test].values)

		y_train = target_copy[train]
		y_val = target_copy[test]

		#y_train = np.sqrt(y_train + 1)

		if 'NeuralNet' in name:
			if params['num_layers'] == 2:
				model = NeuralNetworkModel(n_features=X_train.shape[1],
							optimizer=params['optimizer_2'],
							#dropout=params['dropout_2'],
							activations=(params['activations_21'], params['activations_22']),
							nodes=(params['nodes_21'], params['nodes_22']),
							loss=params['loss_2'],
							batch_size=params['batch_2'],
							metrics=['mse', 'mae'])

				history = model.fit(X_train, y_train, X_val, y_val,
								   		verbose=0, 
								   		epochs=500)

			elif params['num_layers'] == 3:
				model = NeuralNetworkModel(n_features=X_train.shape[1],
							optimizer=params['optimizer_3'],
							#dropout=params['dropout_3'],
							activations=(params['activations_31'], params['activations_32'], params['activations_33']),
							nodes=(params['nodes_31'], params['nodes_32'], params['nodes_33']),
							loss=params['loss_3'],
							batch_size=params['batch_3'], 
							metrics=['mse', 'mae'])

				history = model.fit(X_train, y_train, X_val, y_val,
								   		verbose=0,						   		
								   		epochs=500)

			else:
				model = NeuralNetworkModel(n_features=X_train.shape[1],
							optimizer=params['optimizer_4'],
							#dropout=params['dropout_4'],
							activations=(params['activations_41'], params['activations_42'], params['activations_43'], params['activations_44']),
							nodes=(params['nodes_41'], params['nodes_42'], params['nodes_43'], params['nodes_44']),
							loss=params['loss_4'],
							batch_size=params['batch_4'], 
							metrics=['mse', 'mae'])

				history = model.fit(X_train, y_train, X_val, y_val,
								   		verbose=0,
								   		epochs=500)

			# y_pred = np.power(model.predict(X_val), 2)-1
			y_pred = model.predict(X_val)
			y_pred = y_pred.clip(min = .01)
		else:
			if 'CatBoost' in name:
				model = CatBoostRegressor(objective=params['objective'],
									learning_rate=params['learning_rate'],
									#l2_leaf_reg=score1_params['l2_leaf_reg'],
									max_depth=params['max_depth'],
									colsample_bylevel=params['colsample_bylevel'],
									bagging_temperature=params['bagging_temperature'],
									random_strength=params['random_strength'],
									verbose=0)
				model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)

			elif 'XGBoost' in name:
				model = XGBRegressor(n_estimators=params['n_estimators'],
											objective=params['objective'],
											eta=params['eta'],
											max_depth=params['max_depth'],
											min_child_weight=params['min_child_weight'],
											subsample=params['subsample'],
											gamma=params['gamma'],
											colsample_bytree=params['colsample_bytree'],
											#reg_lambda=params['lambda'],
											nthread=4,
											booster='gbtree',
											tree_method='exact')
				model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

			elif 'HistGradientBoost' in name:
				model = HistGradientBoostingRegressor(loss='poisson',
													learning_rate=params['learning_rate'],
													max_leaf_nodes=params['max_leaf_nodes'],
													min_samples_leaf=params['min_samples_leaf'])
				model.fit(X_train, y_train)

			else:
				model = ExtraTreesRegressor(n_estimators=params['n_estimators'],
											min_samples_split=params['min_samples_split'],
											min_samples_leaf=params['min_samples_leaf'],
											max_features=params['max_features'])
				model.fit(X_train, y_train)

			#y_pred = np.power(model.predict(X_val), 2)-1
			y_pred = model.predict(X_val)
			y_pred = y_pred.clip(min = .01)

		# foot_poisson = FootballPoissonModel()
		# home_win, draw, away_win = foot_poisson.predict_chances(score1_pred, score2_pred)
		# over, under = foot_poisson.predict_overs(score1_pred, score2_pred)
		# df_predictions = df_train.iloc[test].copy()

		# predictions = pd.DataFrame(data={'score1_pred': score1_pred, 'score2_pred': score2_pred,
		# 								 'homewin_pred': np.clip(list(home_win), a_min=0.01, a_max=None), 'draw_pred': np.clip(list(draw), a_min=0.01, a_max=None), 'awaywin_pred': np.clip(list(away_win), a_min=0.01, a_max=None),
		# 								 '>2.5_pred': over, '<2.5_pred': under})
		# df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
		# book = Bookmaker(df_predictions, odds='max', stake=5)
		# book.calculate()

		mse.append(mean_squared_error(y_val, y_pred))
		mae.append(mean_absolute_error(y_val, y_pred))
		pdev.append(mean_poisson_deviance(y_val, y_pred))
		rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
		mape.append(mean_absolute_percentage_error(y_val, y_pred))
		msle.append(mean_squared_log_error(y_val, y_pred))
		medae.append(median_absolute_error(y_val, y_pred))
		r2.append(r2_score(y_val, y_pred))
		huber.append(huber_loss(y_val, y_pred))
		poisson.append(poisson_loss(y_val, y_pred))
		#logcosh.append(logcosh_loss(y_val, y_pred))

	loss = np.mean(poisson)

	return {'status': 'ok',
			'loss': loss,
			'pdev': np.mean(pdev),
			'rmse': np.mean(rmse),
			'mae': np.mean(mae),
			'mape': np.mean(mape),
			'msle': np.mean(msle),
			'medae': np.mean(medae),
			'r2': np.mean(r2),
			'params': params,
			'model': model}




pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
df = pd.read_csv('training_data\\training_data_decay00325_num40_wluck.csv')
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df = df.rename(columns={'matchday': 'matchday_home'})
df.dropna(inplace=True)

df.drop(['weight', 'weight_away'], axis=1, inplace=True)

# df = df[df['league_id'] == 1845]

test_matches_number = 4000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()
other_columns = ['team1', 'team2', 'spi1', 'spi2', 'importance1', 'importance2', 'score1', 'score2',
				'score1_similar', 'score2_similar', 'xg1_similar', 'xg2_similar',
				'score1_league', 'score2_league', 'xg1_league', 'xg2_league', 'A', 'H']
home_stats_cols = [col for col in df_train.columns.values if 'home' in col]
home_other_cols = [col for col in other_columns if '1' in col or 'H' == col]
away_stats_cols = [col for col in df_train.columns.values if 'away' in col]
away_other_cols = [col for col in other_columns if '2' in col or 'A' == col]
rename_away_stats_dict = dict(zip(home_stats_cols + home_other_cols + away_stats_cols + away_other_cols, 
								  away_stats_cols + away_other_cols + home_stats_cols + home_other_cols))

df_train_transformed = pd.concat([df_train.assign(home=1),
					  df_train.assign(home=0).rename(columns=rename_away_stats_dict)], sort=False, ignore_index=True)

df_test_transformed = pd.concat([df_test.assign(home=1),
					  df_test.assign(home=0).rename(columns=rename_away_stats_dict)], sort=False, ignore_index=True).copy()


df_train_transformed['spi_diff'] = df_train_transformed['spi1'] - df_train_transformed['spi2']
df_train_transformed['importance_diff'] = df_train_transformed['importance1'] - df_train_transformed['importance2']

df_train_transformed['adj_avg_xg1_diff'] = df_train_transformed['adj_avg_xg1_home'] - df_train_transformed['adj_avg_xg1_away']
df_train_transformed['adj_avg_xg2_diff'] = df_train_transformed['adj_avg_xg2_home'] - df_train_transformed['adj_avg_xg2_away']
df_train_transformed['adj_avg_xg_diff_home'] = df_train_transformed['adj_avg_xg1_home'] - df_train_transformed['adj_avg_xg2_home']
df_train_transformed['adj_avg_xg_diff_away'] = df_train_transformed['adj_avg_xg1_away'] - df_train_transformed['adj_avg_xg2_away']

df_train_transformed['shotsot1_diff'] = df_train_transformed['shotsot1_home'] - df_train_transformed['shotsot1_away']
df_train_transformed['shotsot2_diff'] = df_train_transformed['shotsot2_home'] - df_train_transformed['shotsot2_away']
df_train_transformed['shotsot_diff_home'] = df_train_transformed['shotsot1_home'] - df_train_transformed['shotsot2_home']
df_train_transformed['shotsot_diff_away'] = df_train_transformed['shotsot1_away'] - df_train_transformed['shotsot2_away']

df_train_transformed['corners1_diff'] = df_train_transformed['corners1_home'] - df_train_transformed['corners1_away']
df_train_transformed['corners2_diff'] = df_train_transformed['corners2_home'] - df_train_transformed['corners2_away']

df_train_transformed['xpts1_diff'] = df_train_transformed['xpts1_home'] - df_train_transformed['xpts1_away']

df_train_transformed['xgshot1_diff'] = df_train_transformed['xgshot1_home'] - df_train_transformed['xgshot1_away']
df_train_transformed['convrate1_diff'] = df_train_transformed['convrate1_home'] - df_train_transformed['convrate1_away']

df_train_transformed['month'] = pd.to_datetime(df_train_transformed['date']).dt.month
bins = [0, 10, 20, 30, 40, 50]
labels = ['<10', '11-20', '21-30', '31-40', '41-50']
df_train_transformed['matchday_binned'] = pd.cut(df_train_transformed['matchday_home'], bins=bins, labels=labels)


df_test_transformed['spi_diff'] = df_test_transformed['spi1'] - df_test_transformed['spi2']
df_test_transformed['importance_diff'] = df_test_transformed['importance1'] - df_test_transformed['importance2']

df_test_transformed['adj_avg_xg1_diff'] = df_test_transformed['adj_avg_xg1_home'] - df_test_transformed['adj_avg_xg1_away']
df_test_transformed['adj_avg_xg2_diff'] = df_test_transformed['adj_avg_xg2_home'] - df_test_transformed['adj_avg_xg2_away']
df_test_transformed['adj_avg_xg_diff_home'] = df_test_transformed['adj_avg_xg1_home'] - df_test_transformed['adj_avg_xg2_home']
df_test_transformed['adj_avg_xg_diff_away'] = df_test_transformed['adj_avg_xg1_away'] - df_test_transformed['adj_avg_xg2_away']

df_test_transformed['shotsot1_diff'] = df_test_transformed['shotsot1_home'] - df_test_transformed['shotsot1_away']
df_test_transformed['shotsot2_diff'] = df_test_transformed['shotsot2_home'] - df_test_transformed['shotsot2_away']
df_test_transformed['shotsot_diff_home'] = df_test_transformed['shotsot1_home'] - df_test_transformed['shotsot2_home']
df_test_transformed['shotsot_diff_away'] = df_test_transformed['shotsot1_away'] - df_test_transformed['shotsot2_away']

df_test_transformed['corners1_diff'] = df_test_transformed['corners1_home'] - df_test_transformed['corners1_away']
df_test_transformed['corners2_diff'] = df_test_transformed['corners2_home'] - df_test_transformed['corners2_away']

df_test_transformed['xpts1_diff'] = df_test_transformed['xpts1_home'] - df_test_transformed['xpts1_away']

df_test_transformed['xgshot1_diff'] = df_test_transformed['xgshot1_home'] - df_test_transformed['xgshot1_away']
df_test_transformed['convrate1_diff'] = df_test_transformed['convrate1_home'] - df_test_transformed['convrate1_away']

df_test_transformed['month'] = pd.to_datetime(df_test_transformed['date']).dt.month
df_test_transformed['matchday_binned'] = pd.cut(df_test_transformed['matchday_home'], bins=bins, labels=labels)


numerical_features = [
			'adj_avg_xg1_home',
			'adj_avg_xg2_away',
			'score1_similar',
			'importance_diff', 
			'shotsot1_diff',
			'shotsot2_diff',
			'corners1_diff',
			'corners2_diff',
			]



categorical_features = [
			# 'home',
			# 'month_1',
			# 'month_2',
			# 'month_3',
			# 'month_4',
			# 'month_5',
			# 'month_7',
			# 'month_8',
			# 'month_9',
			# 'month_10',
			# 'month_11',
			# 'month_12',
			# 'matchday_<10', 'matchday_11-20', 'matchday_21-30', 'matchday_31-40', 
			# 'matchday_41-50',
			# #'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3',
			# 'league_1843.0', 'league_1845.0', 'league_1846.0', 'league_1854.0', 'league_1856.0',
			# 'league_1864.0', 'league_1869.0', 'league_2411.0', 
			# 'league_2412.0'
			]


target_train_transformed = df_train_transformed.score1.values

print(f'Number of NA values: {df_train_transformed.isna().sum().sum()}')
print(f'Number of INF values: {df_train_transformed.replace([np.inf, -np.inf], np.nan).isna().sum().sum()}')

kmeans_kwargs = {
		"init": "random",
		"n_init": 10,
		"max_iter": 300,
		"random_state": 42,
		}


train_scaled = df_train_transformed[numerical_features]
# transformer = FeatureTransformer()
# transformer.fit(df_train_transformed)
# df_train_strght = transformer.transform(df_train_transformed)


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_scaled.values)

# sse = []
# for k in range(1, 11):
# 	kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
# 	kmeans.fit(scaled)
# 	sse.append(kmeans.inertia_)

# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()

# kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
# print(kl.elbow)

kmeans = KMeans(n_clusters=4, **kmeans_kwargs)
kmeans.fit(train_scaled)
df_train_transformed['cluster'] = kmeans.labels_
one_hot = pd.get_dummies(df_train_transformed['cluster'], prefix='cluster')
df_train_transformed = df_train_transformed.drop('cluster',axis = 1)
df_train_transformed = df_train_transformed.join(one_hot)


one_hot = pd.get_dummies(df_train_transformed['league_id'], prefix='league')
df_train_transformed = df_train_transformed.drop('league_id',axis = 1)
df_train_transformed = df_train_transformed.join(one_hot)

one_hot = pd.get_dummies(df_train_transformed['matchday_binned'], prefix='matchday')
df_train_transformed = df_train_transformed.drop('matchday_binned',axis = 1)
df_train_transformed = df_train_transformed.join(one_hot)

one_hot = pd.get_dummies(df_train_transformed['month'], prefix='month')
df_train_transformed = df_train_transformed.drop('month',axis = 1)
df_train_transformed = df_train_transformed.join(one_hot)

test_scaled = scaler.transform(df_test_transformed[numerical_features].values)
df_test_transformed['cluster'] = kmeans.predict(test_scaled)
one_hot = pd.get_dummies(df_test_transformed['cluster'], prefix='cluster')
df_test_transformed = df_test_transformed.drop('cluster', axis = 1)
df_test_transformed = df_test_transformed.join(one_hot)

one_hot = pd.get_dummies(df_test_transformed['league_id'], prefix='league')
df_test_transformed = df_test_transformed.drop('league_id',axis = 1)
df_test_transformed = df_test_transformed.join(one_hot)

one_hot = pd.get_dummies(df_test_transformed['matchday_binned'], prefix='matchday')
df_test_transformed = df_test_transformed.drop('matchday_binned',axis = 1)
df_test_transformed = df_test_transformed.join(one_hot)

one_hot = pd.get_dummies(df_test_transformed['month'], prefix='month')
df_test_transformed = df_test_transformed.drop('month',axis = 1)
df_test_transformed = df_test_transformed.join(one_hot)




try:
	scope.define(tf.keras.optimizers.Adam)
	scope.define(tf.keras.optimizers.Adagrad)
	scope.define(tf.keras.optimizers.RMSprop)
	scope.define(tf.keras.optimizers.Nadam)
	scope.define(tf.keras.optimizers.SGD)
except ValueError:
	pass

neuralnet_space_mae = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 1256, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['mae']),
								'features': hp.choice('features_2', [numerical_features]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.002)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 1256, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['mae']),
								'features': hp.choice('features_3', [numerical_features]),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.002)))
									])}
				])

neuralnet_space_poisson = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 1256, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['poisson']),
								'features': hp.choice('features_2', [numerical_features]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.002)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 1256, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['poisson']),
								'features': hp.choice('features_3', [numerical_features]),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.002)))
									])}
				])

neuralnet_space_logcosh = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 1256, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['logcosh']),
								'features': hp.choice('features_2', [numerical_features]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.002)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 1256, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['logcosh']),
								'features': hp.choice('features_3', [numerical_features]),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.002)))
									])}
				])

neuralnet_space_huber = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 1256, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['huber_loss']),
								'features': hp.choice('features_2', [numerical_features]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.002)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 1256, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['huber_loss']),
								'features': hp.choice('features_3', [numerical_features]),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.002)))
									])}
				])

neuralnet_space_mape = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 1256, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['mean_absolute_percentage_error']),
								'features': hp.choice('features_2', [numerical_features]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.002)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 1256, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['mean_absolute_percentage_error']),
								'features': hp.choice('features_3', [numerical_features]),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.002))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.002)))
									])}
				])

neuralnet_space_mse = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 700, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['mse']),
								'features': hp.choice('features_2', [numerical_features]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.0012))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.0012))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.0012))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.0012))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.0012)))
									])}
								])


xgb_space_sqrd_error = {
		'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
		'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
		'max_depth':  scope.int(hp.quniform('max_depth', 2, 20, 1)),
		'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1)),
		'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
		'gamma': hp.quniform('gamma', 0.5, 3, 0.05),
		'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
		'features': hp.choice('features', [numerical_features]),
		'objective': hp.choice('objective', ['reg:squarederror'])
		#'lambda': hp.quniform('lambda', 0, 5, 0.1),
		#'alpha': hp.quniform('alpha', 0, 3, 0.1),
		}

xgb_space_poisson = {
		'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
		'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
		'max_depth':  scope.int(hp.quniform('max_depth', 2, 20, 1)),
		'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1)),
		'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
		'gamma': hp.quniform('gamma', 0.5, 3, 0.05),
		'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
		'features': hp.choice('features', [numerical_features]),
		'objective': hp.choice('objective', ['count:poisson'])
		#'lambda': hp.quniform('lambda', 0, 5, 0.1),
		#'alpha': hp.quniform('alpha', 0, 3, 0.1),
		}


catboost_space_huber = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		#'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 5.0),
		'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
		'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1.0),
		'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
		'random_strength': hp.uniform('random_strength', 0.0, 100),
		'objective': hp.choice('objective', ['Huber:delta=1']),
		'features': hp.choice('features', [numerical_features])
		# 'pca_n_components': scope.int(hp.quniform('pca_n_components', 2, 30, 1))
		}

catboost_space_mae = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		#'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 5.0),
		'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
		'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1.0),
		'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
		'random_strength': hp.uniform('random_strength', 0.0, 100),
		'objective': hp.choice('objective', ['MAE']),
		'features': hp.choice('features', [numerical_features])
		# 'pca_n_components': scope.int(hp.quniform('pca_n_components', 2, 30, 1))
		}

catboost_space_poisson = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		#'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 5.0),
		'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
		'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1.0),
		'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
		'random_strength': hp.uniform('random_strength', 0.0, 100),
		'objective': hp.choice('objective', ['Poisson']),
		'features': hp.choice('features', [numerical_features])
		# 'pca_n_components': scope.int(hp.quniform('pca_n_components', 2, 30, 1))
		}


extratrees_space = {
		'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
		'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
		'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
		'features': hp.choice('features', [numerical_features])
		}


histgradboost_space = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		'max_leaf_nodes': scope.int(hp.quniform('max_leaf_nodes', 2, 60, 1)),
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 60, 1)),
		'features': hp.choice('features', [numerical_features])
		}


models = {#'XGBoost_sqrd_err': xgb_space_sqrd_error,
			# 'XGBoost_poisson': xgb_space_poisson,
			# 'CatBoost_huber': catboost_space_huber,
			# 'CatBoost_mae': catboost_space_mae,
			# 'CatBoost_poisson': catboost_space_poisson,
			# 'NeuralNet_mae': neuralnet_space_mae,
			'NeuralNet_poisson': neuralnet_space_poisson,
			# 'NeuralNet_logcosh': neuralnet_space_logcosh,
			# 'NeuralNet_huber': neuralnet_space_huber,
			# 'NeuralNet_mape': neuralnet_space_mape,
			# 'NeuralNet_mse': neuralnet_space_mse,
			# 'HistGradientBoost': histgradboost_space,
			# 'ExtraTrees': extratrees_space	
			}


basic_models = {
		'Dummy_mean': DummyRegressor(strategy = "mean"),
		'Dummy_median': DummyRegressor(strategy = "median"),
		'Poisson_reg': PoissonRegressor(max_iter = 5000),
		'XGB_poisson': XGBRegressor(objective = "count:poisson"),
		'XGB_mae': XGBRegressor(objective = "reg:squarederror"),
		'HistGradientBoost': HistGradientBoostingRegressor(loss = "poisson"),
		'ExtraTrees': ExtraTreesRegressor(),
		'CatBoost_poisson':CatBoostRegressor(objective = "Poisson", verbose = 0),
		'CatBoost_mae': CatBoostRegressor(objective = "MAE", verbose = 0),
		'CatBoost_huber': CatBoostRegressor(objective = "Huber:delta=200", verbose = 0),
		'NeuralNet_poisson': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='poisson'),
		'NeuralNet_logcosh': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='logcosh'),
		'NeuralNet_mae': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='mean_absolute_error'),
		# 'NeuralNet_mse': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='mean_squared_error'),
		# 'NeuralNet_msle': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='mean_squared_logarithmic_error'),
		# 'NeuralNet_mape': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='mean_absolute_percentage_error'),
		# 'NeuralNet_huber': NeuralNetworkModel(n_features=len(numerical_features+categorical_features), loss='huber_loss')
		}



catboost_mae_params = {'bagging_temperature': 69.32347629336867, 'colsample_bylevel': 0.9042263919423306, 'learning_rate': 0.059135216445649785, 'max_depth': 6, 'objective': 'MAE', 'random_strength': 57.5044030038614}
neuralnet_mae_params = {'activations': ('hard_sigmoid', 'selu', 'tanh'), 'batch': 60, 'loss': 'mae', 'nodes': (14, 44, 104), 'num_layers': 3, 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0010012101847678423)}
xgboost_poisson_params = {'colsample_bytree': 0.9, 'eta': 0.2, 'gamma': 1.25, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 885, 'objective': 'count:poisson', 'subsample': 0.8}
catboost_poisson_params = {'bagging_temperature': 10.45943130929077, 'colsample_bylevel': 0.4316079607705634, 'learning_rate': 0.1811321733393647, 'max_depth': 4, 'objective': 'Poisson', 'random_strength': 38.80202988079048}
neuralnet_poisson_params = {'activations': ('sigmoid', 'sigmoid'), 'batch': 994, 'loss': 'poisson', 'nodes': (2, 306), 'num_layers': 2, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001287)}
histgradboost_poisson_params = {'learning_rate': 0.10109485695245424, 'max_leaf_nodes': 3, 'min_samples_leaf': 57}
xgboost_sqrd_err_params = {'colsample_bytree': 0.75, 'eta': 0.25, 'gamma': 1.4000000000000001, 'max_depth': 3, 'min_child_weight': 4, 'n_estimators': 291, 'objective': 'reg:squarederror', 'subsample': 0.8500000000000001}
neuralnet_mse_params =  {'activations': ('sigmoid', 'tanh'), 'batch': 308, 'loss': 'mse', 'nodes': (52, 18), 'num_layers': 2, 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)}
extra_trees_mse_params = {'max_features': 'log2', 'min_samples_leaf': 7, 'min_samples_split': 7, 'n_estimators': 478}
catboost_huber_params = {'bagging_temperature': 8.06887605641524, 'colsample_bylevel': 0.4750844452806466, 'learning_rate': 0.08477793380600201, 'max_depth': 4, 'objective': 'Huber:delta=1', 'random_strength': 91.32730657879162}
neuralnet_huber_params = {'activations': ('sigmoid', 'hard_sigmoid', 'sigmoid'), 'batch': 1002, 'loss': 'huber_loss', 'nodes': (4, 446, 20), 'num_layers': 3, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001046389)}


opt_models = {
				'CatBoost_mae': CatBoostRegressor(objective=catboost_mae_params['objective'],
								learning_rate=catboost_mae_params['learning_rate'],
								#l2_leaf_reg=score1_params['l2_leaf_reg'],
								max_depth=catboost_mae_params['max_depth'],
								colsample_bylevel=catboost_mae_params['colsample_bylevel'],
								bagging_temperature=catboost_mae_params['bagging_temperature'],
								random_strength=catboost_mae_params['random_strength'],
								verbose=0),
			 # 'CatBoost_poisson': CatBoostRegressor(objective=catboost_poisson_params['objective'],
				# 				learning_rate=catboost_poisson_params['learning_rate'],
				# 				#l2_leaf_reg=score1_params['l2_leaf_reg'],
				# 				max_depth=catboost_poisson_params['max_depth'],
				# 				colsample_bylevel=catboost_poisson_params['colsample_bylevel'],
				# 				bagging_temperature=catboost_poisson_params['bagging_temperature'],
				# 				random_strength=catboost_poisson_params['random_strength'],
				# 				verbose=0),
			  'XGB_poisson': XGBRegressor(n_estimators=xgboost_poisson_params['n_estimators'],
							eta=xgboost_poisson_params['eta'],
							max_depth=xgboost_poisson_params['max_depth'],
							min_child_weight=xgboost_poisson_params['min_child_weight'],
							subsample=xgboost_poisson_params['subsample'],
							gamma=xgboost_poisson_params['gamma'],
							colsample_bytree=xgboost_poisson_params['colsample_bytree'],
							#reg_lambda=xgb_params['lambda'],
							nthread=4,
							booster='gbtree',
							tree_method='exact'),
			# 'XGB_mse': XGBRegressor(n_estimators=xgboost_sqrd_err_params['n_estimators'],
			# 				eta=xgboost_sqrd_err_params['eta'],
			# 				max_depth=xgboost_sqrd_err_params['max_depth'],
			# 				min_child_weight=xgboost_sqrd_err_params['min_child_weight'],
			# 				subsample=xgboost_sqrd_err_params['subsample'],
			# 				gamma=xgboost_sqrd_err_params['gamma'],
			# 				colsample_bytree=xgboost_sqrd_err_params['colsample_bytree'],
			# 				#reg_lambda=xgb_params['lambda'],
			# 				nthread=4,
			# 				booster='gbtree',
			# 				tree_method='exact'),
			  # 'HistGradBoost': HistGradientBoostingRegressor(loss='poisson',
					# 						learning_rate=histgradboost_poisson_params['learning_rate'],
					# 						max_leaf_nodes=histgradboost_poisson_params['max_leaf_nodes'],
					# 						min_samples_leaf=histgradboost_poisson_params['min_samples_leaf']),
			  # 'ExtraTrees': ExtraTreesRegressor(n_estimators=extra_trees_mse_params['n_estimators'],
					# 						min_samples_split=extra_trees_mse_params['min_samples_split'],
					# 						min_samples_leaf=extra_trees_mse_params['min_samples_leaf'],
					# 						max_features=extra_trees_mse_params['max_features']),
			  'NeuralNet_mae': NeuralNetworkModel(n_features=len(numerical_features+categorical_features),
							optimizer=neuralnet_mae_params['optimizer'],
							#dropout=params['dropout_3'],
							activations=neuralnet_mae_params['activations'],
							nodes=neuralnet_mae_params['nodes'],
							loss=neuralnet_mae_params['loss'],
							batch_size=neuralnet_mae_params['batch'],
							metrics=['mse', 'mae']),
			  # 'NeuralNet_Poisson': NeuralNetworkModel(n_features=len(numerical_features+categorical_features),
					# 		optimizer=neuralnet_poisson_params['optimizer'],
					# 		#dropout=params['dropout_3'],
					# 		activations=neuralnet_poisson_params['activations'],
					# 		nodes=neuralnet_poisson_params['nodes'],
					# 		loss=neuralnet_poisson_params['loss'],
					# 		batch_size=neuralnet_poisson_params['batch'],
					# 		metrics=['mse', 'mae']),
			  # 'NeuralNet_mse': NeuralNetworkModel(n_features=len(numerical_features+categorical_features),
					# 		optimizer=neuralnet_mse_params['optimizer'],
					# 		#dropout=params['dropout_3'],
					# 		activations=neuralnet_mse_params['activations'],
					# 		nodes=neuralnet_mse_params['nodes'],
					# 		loss=neuralnet_mse_params['loss'],
					# 		batch_size=neuralnet_poisson_params['batch'],
					# 		metrics=['mse', 'mae']),
			  # 'NeuralNet_huber': NeuralNetworkModel(n_features=len(numerical_features+categorical_features),
					# 		optimizer=neuralnet_huber_params['optimizer'],
					# 		#dropout=params['dropout_3'],
					# 		activations=neuralnet_huber_params['activations'],
					# 		nodes=neuralnet_huber_params['nodes'],
					# 		loss=neuralnet_huber_params['loss'],
					# 		batch_size=neuralnet_huber_params['batch'],
					# 		metrics=['mse', 'mae']),
			  # 'NeuralNet_mape': NeuralNetworkModel(n_features=len(numerical_features+categorical_features),
					# 		optimizer=neuralnet_mape_params['optimizer'],
					# 		#dropout=params['dropout_3'],
					# 		activations=neuralnet_mape_params['activations'],
					# 		nodes=neuralnet_mape_params['nodes'],
					# 		loss=neuralnet_mape_params['loss'],
					# 		batch_size=neuralnet_mape_params['batch'],
					# 		metrics=['mse', 'mae']),
			  # 'NeuralNet_logcosh': NeuralNetworkModel(n_features=len(numerical_features+categorical_features),
					# 		optimizer=neuralnet_logcosh_params['optimizer'],
					# 		#dropout=params['dropout_3'],
					# 		activations=neuralnet_logcosh_params['activations'],
					# 		nodes=neuralnet_logcosh_params['nodes'],
					# 		loss=neuralnet_logcosh_params['loss'],
					# 		batch_size=neuralnet_logcosh_params['batch'],
					# 		metrics=['mse', 'mae'])
			  }


# res_dict = {}
# for model_name, model in basic_models.items():
# 	mse, mae, pdev, rmse, mse_test, mae_test, pdev_test, rmse_test, mape_test, msle_test, medae_test, r2_test, returns, returns_over = evaluate(model, model_name, features=numerical_features+categorical_features, df_train=df_train_transformed, target=target_train_transformed, 
# 																							df_test=df_test_transformed, cv=4)
# 	res_dict[model_name] = [mse, mae, pdev, rmse, mse_test, mae_test, pdev_test, rmse_test, returns, returns_over]
# results_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['MSE_train', 'MAE_train', 'P_DEVIANCE_train', 'RMSE_train', 'MSE_test', 'MAE_test', 'P_DEVIANCE_test', 'RMSE_test', 'Return', 'Return Over'])
# print(results_df.sort_values(by='Return'))


foot_poisson = FootballPoissonModel()
score1_pred_avg = np.zeros(df_test.shape[0])
score2_pred_avg = np.zeros(df_test.shape[0])
res_dict = {}
for model_name, model in opt_models.items():
	score1_true, score2_true, score1_pred, score2_pred = evaluate(model, model_name, features=numerical_features+categorical_features, 
															df_train=df_train_transformed, 
															target=target_train_transformed,
															df_test=df_test_transformed,
															cv=4, 
															#save=['NeuralNet_mae', 'XGB_poisson', 'CatBoost_mae']
															)


	mse_test = (mean_squared_error(score1_true, score1_pred) + mean_squared_error(score2_true, score2_pred)) /2
	mae_test = (mean_absolute_error(score1_true, score1_pred) + mean_absolute_error(score2_true, score2_pred)) /2
	pdev_test = (mean_poisson_deviance(score1_true, score1_pred) + mean_poisson_deviance(score2_true, score2_pred)) /2
	rmse_test = (np.sqrt(mean_squared_error(score1_true, score1_pred)) + np.sqrt(mean_squared_error(score2_true, score2_pred))) /2
	mape_test = (mean_absolute_percentage_error(score1_true, score1_pred) + mean_absolute_percentage_error(score2_true, score2_pred)) /2
	msle_test = (mean_squared_log_error(score1_true, score1_pred) + mean_squared_log_error(score2_true, score2_pred)) /2
	medae_test = (median_absolute_error(score1_true, score1_pred) + median_absolute_error(score2_true, score2_pred)) /2
	r2_test = (r2_score(score1_true, score1_pred) + r2_score(score2_true, score2_pred)) /2



	score1_pred_avg = score1_pred_avg + score1_pred.ravel()
	score2_pred_avg = score2_pred_avg + score2_pred.ravel()

	home_win, draw, away_win = foot_poisson.predict_chances(score1_pred, score2_pred)
	over, under = foot_poisson.predict_overs(score1_pred, score2_pred)
	df_predictions = df_test.copy()

	predictions = pd.DataFrame(data={'score1_pred': score1_pred.ravel(), 'score2_pred': score2_pred.ravel(),
								 'homewin_pred': np.clip(list(home_win), a_min=0.01, a_max=None), 
								 'draw_pred': np.clip(list(draw), a_min=0.01, a_max=None), 
								 'awaywin_pred': np.clip(list(away_win), a_min=0.01, a_max=None),
								 '>2.5_pred': over, '<2.5_pred': under})

	df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
	book = Bookmaker(df_predictions, odds='max', stake=5)
	book.calculate()

	#df_predictions.to_csv(f'{model_name}_predictions.csv')

	returns = df_predictions[['bet_return']].sum().values
	returns_over = df_predictions[['bet_return_over']].sum().values
	
	res_dict[model_name] = [mse_test, mae_test, pdev_test, rmse_test, mape_test, msle_test, medae_test, r2_test, returns, returns_over]

score1_pred_avg = score1_pred_avg/len(opt_models)
score2_pred_avg = score2_pred_avg/len(opt_models)


mse_test = (mean_squared_error(score1_true, score1_pred_avg) + mean_squared_error(score2_true, score2_pred_avg)) /2
mae_test = (mean_absolute_error(score1_true, score1_pred_avg) + mean_absolute_error(score2_true, score2_pred_avg)) /2
pdev_test = (mean_poisson_deviance(score1_true, score1_pred_avg) + mean_poisson_deviance(score2_true, score2_pred_avg)) /2
rmse_test = (np.sqrt(mean_squared_error(score1_true, score1_pred_avg)) + np.sqrt(mean_squared_error(score2_true, score2_pred_avg))) /2
mape_test = (mean_absolute_percentage_error(score1_true, score1_pred_avg) + mean_absolute_percentage_error(score2_true, score2_pred_avg)) /2
msle_test = (mean_squared_log_error(score1_true, score1_pred_avg) + mean_squared_log_error(score2_true, score2_pred_avg)) /2
medae_test = (median_absolute_error(score1_true, score1_pred_avg) + median_absolute_error(score2_true, score2_pred_avg)) /2
r2_test = (r2_score(score1_true, score1_pred_avg) + r2_score(score2_true, score2_pred_avg)) /2

home_win, draw, away_win = foot_poisson.predict_chances(score1_pred_avg, score2_pred_avg)
over, under = foot_poisson.predict_overs(score1_pred_avg, score2_pred_avg)
df_predictions = df_test.copy()

predictions = pd.DataFrame(data={'score1_pred': score1_pred_avg.ravel(), 'score2_pred': score2_pred_avg.ravel(),
							 'homewin_pred': np.clip(list(home_win), a_min=0.01, a_max=None), 
							 'draw_pred': np.clip(list(draw), a_min=0.01, a_max=None), 
							 'awaywin_pred': np.clip(list(away_win), a_min=0.01, a_max=None),
							 '>2.5_pred': over, '<2.5_pred': under})

df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
book = Bookmaker(df_predictions, odds='max', stake=5)
book.calculate()

#df_predictions.to_csv(f'{model_name}_predictions.csv')

returns = df_predictions[['bet_return']].sum().values
returns_over = df_predictions[['bet_return_over']].sum().values

res_dict['average_model'] = [mse_test, mae_test, pdev_test, rmse_test, mape_test, msle_test, medae_test, r2_test, returns, returns_over]

results_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['MSE_test', 'MAE_test', 'P_DEVIANCE_test', 'RMSE_test', 'MAPE_Test', 'MSLE_test', 'MedAE_test', 'r2_test', 'Return', 'Return Over'])
print(results_df.sort_values(by='Return'))


# res_dict = {}
# for model, space in models.items():
# 	print(model)
# 	opt_f = partial(optimize, name=model, df_train=df_train_transformed, df_target=target_train_transformed)
# 	trials = Trials()
# 	best = fmin(fn=opt_f,
# 	            space=space,
# 	            algo=tpe.suggest,
# 	            max_evals=50,
# 	            trials=trials)

# 	best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
# 	print(f'Best model params: {best_params}')
# 	best_loss = trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
# 	print(f'Best avg loss: {best_loss}')
# 	best_pdev = trials.results[np.argmin([r['loss'] for r in trials.results])]['pdev']
# 	print(f'Best avg Poisson deviance: {best_pdev}')
# 	best_rmse = trials.results[np.argmin([r['loss'] for r in trials.results])]['rmse']
# 	print(f'Best avg RMSE: {best_rmse}')
# 	best_mae = trials.results[np.argmin([r['loss'] for r in trials.results])]['mae']
# 	print(f'Best mae: {best_mae}')
# 	best_mape = trials.results[np.argmin([r['loss'] for r in trials.results])]['mape']
# 	print(f'Best avg mape: {best_mape}')
# 	best_msle = trials.results[np.argmin([r['loss'] for r in trials.results])]['msle']
# 	print(f'Best avg msle: {best_msle}')
# 	best_r2 = trials.results[np.argmin([r['loss'] for r in trials.results])]['r2']
# 	print(f'Best avg r2: {best_r2}')
# 	best_medae = trials.results[np.argmin([r['loss'] for r in trials.results])]['medae']
# 	print(f'Best medae: {best_medae}')
# 	best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
# 	if model in ['NeuralNet_mae', 'NeuralNet_mse' 'NeuralNet_poisson', 'NeuralNet_logcosh', 'NeuralNet_huber', 'NeuralNet_mape']:
# 		print(f'Best learning rate" {best_model.model.optimizer.lr.numpy()}')

# 	res_dict[model] = [best_loss, best_pdev, best_rmse, best_mae, best_mape, best_msle, best_r2, best_medae]

# results_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['loss', 'P DEVIANCE', 'RMSE', 'MAE', 'MAPE', 'MSLE', 'R2', 'MedAE'])
# print(results_df.sort_values(by='MAE'))



# nn_model = NeuralNetworkModel(name='NeuralNet_mae')
# cb_model = CatBoostRegressor()
# cb_model.load_model('models//CatBoost_mae', format='cbm')
# xgb_model = XGBRegressor()  # init model
# xgb_model.load_model('models//XGB_poisson')



'''
CatBoost_mae
Best model params: {'bagging_temperature': 78.58534746356862, 'colsample_bylevel': 0.3632609874748783, 'learning_rate': 0.608282521552185, 'max_depth': 2, 'objective': 'MAE', 'random_strength': 12.688356126720745}
Best avg loss: 0.867351229770553
Best avg Poisson deviance: 1.2038539808210058
Best avg RMSE: 1.1953193729045646
Best mae: 0.867351229770553
Best avg mape: 1252948484867544.0
Best avg msle: 0.25402101798070387
Best avg r2: 0.05111940213875607
Best medae: 0.9999703517620214

NeuralNet_mae
Best model params: {'activations': ('sigmoid', 'hard_sigmoid', 'hard_sigmoid'), 'batch_3': 292,  'loss': 'mae', 'nodes': (2, 8, 4), 'num_layers': 3, 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)}
Best avg loss: 0.8636511734218888
Best avg Poisson deviance: 1.1837916769059156
Best avg RMSE: 1.1860366454583087
Best mae: 0.8636511734218888
Best avg mape: 1313152364782238.2
Best avg msle: 0.25530659175627635
Best avg r2: 0.06589423652293186
Best medae: 0.9991776645183563

XGBoost_poisson
Best model params: {'colsample_bytree': 0.8500000000000001, 'eta': 0.375, 'gamma': 1.2000000000000002, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 993, 'objective': 'count:poisson', 'subsample': 0.7000000000000001}
Best avg loss: 1.1361005754379323
Best avg Poisson deviance: 1.1361005754379323
Best avg RMSE: 1.1619712963246538
Best mae: 0.8886200443475623
Best avg mape: 1311740655830623.2
Best avg msle: 0.24749297618541732
Best avg r2: 0.10341748112440874
Best medae: 0.8497005999088287

CatBoost_poisson
Best model params: {'bagging_temperature': 47.65845300533181, 'colsample_bylevel': 0.5766841465200845, 'learning_rate': 0.326515365988143, 'max_depth': 1, 'objective': 'Poisson', 'random_strength': 99.53925054053036}
Best avg loss: 1.1290630071474126
Best avg Poisson deviance: 1.1290630071474126
Best avg RMSE: 1.156449369740193
Best mae: 0.8907374573405146
Best avg mape: 1332217516972131.0
Best avg msle: 0.2487818173097863
Best avg r2: 0.11181384450604179
Best medae: 0.8211567303992192

NeuralNet_poisson
Best model params: {'activations': ('hard_sigmoid', 'tanh', 'sigmoid'), 'batch_3': 1254, 'loss_3': 'poisson', 'nodes': (62, 2, 192), 'num_layers': 3, 'optimizer': tf.keras.optimizers.RMSprop(learning_rate=0.001963)}
Best avg loss: 1.121128052985372
Best avg Poisson deviance: 1.121128052985372
Best avg RMSE: 1.1529668833691817
Best mae: 0.893666396423277
Best avg mape: 1397613854957817.5
Best avg msle: 0.2528195686101356
Best avg r2: 0.1171463967136133
Best medae: 0.8329705893993378
Best learning rate" 0.001962782349437475

XGBoost_sqrd_err
Best model params: {'colsample_bytree': 0.75, 'eta': 0.375, 'gamma': 0.8, 'max_depth': 2, 'min_child_weight': 6, 'n_estimators': 613, 'objective': 'reg:squarederror', 'subsample': 0.8500000000000001}
Best avg loss: 1.3484389673504493
Best avg Poisson deviance: 1.138130342712711
Best avg RMSE: 1.1612171213096092
Best mae: 0.8869357717192747
Best avg mape: 1276080467035186.2
Best avg msle: 0.24577748452333098
Best avg r2: 0.10452789878145807
Best medae: 0.8291787207126617

NeuralNet_mse
Best model params: {'activations': ('hard_sigmoid', 'sigmoid'), 'batch': 296, 'loss': 'mse', 'nodes': (64, 36), 'num_layers': 2, 'optimizer': tf.keras.optimizers.RMSprop(learning_rate=0.0012)}
Best avg loss: 1.329077810429873
Best avg Poisson deviance: 1.1235561671959506
Best avg RMSE: 1.1528528841965082
Best mae: 0.887707410717122
Best avg mape: 1333545917050330.0
Best avg msle: 0.2478977464781299
Best avg r2: 0.1173413054115729
Best medae: 0.8271303772926331

CatBoost_huber
Best model params: {'bagging_temperature': 97.0296741395774, 'colsample_bylevel': 0.4298160300802163, 'features': ('adj_avg_xg1_home', 'adj_avg_xg2_away', 'score1_similar', 'importance_diff', 'shotsot1_diff', 'shotsot2_diff', 'corners1_diff', 'corners2_diff'), 'learning_rate': 0.023455863779940515, 'max_depth': 9, 'objective': 'Huber:delta=1', 'random_strength': 70.8981155469218}
Best avg loss: 2.37272310256958
Best avg Poisson deviance: 6.148134808050344
Best avg RMSE: 5.800060153752426
Best mae: 2.7955502560915724
Best avg mape: 3921314837582086.0
Best avg msle: 0.989285886673156
Best avg r2: -23.47520214089249
Best medae: 1.4548152121963231

NeuralNet_huber
Best model params: {'activations': ('hard_sigmoid', 'relu','hard_sigmoid'), 'batch': 738, 'loss': 'huber_loss', 'nodes': (2, 916, '152), 'num_layers': 3, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.00117)}
Best avg loss: 0.564757764339447
Best avg Poisson deviance: 1.2418114406920335
Best avg RMSE: 1.2268414377910333
Best mae: 0.952615528367149
Best avg mape: 1495458982809573.5
Best avg msle: 0.27716714677300885
Best avg r2: 0.0005955347659278576
Best medae: 0.7776321470737457
Best learning rate" 0.001177065190859139

NeuralNet_mape
Best model params: {'activations': ('relu', 'hard_sigmoid'), 'batch': 936, 'loss': 'mean_absolute_percentage_error', 'nodes': (216, 848), 'num_layers': 2, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.00192)}
Best avg loss: 12139883345661.312
Best avg Poisson deviance: 11.974149254338121
Best avg RMSE: 1.8296846347833038
Best mae: 1.3624426083801522
Best avg mape: 12139883345661.312
Best avg msle: 0.7905447695559371
Best avg r2: -1.2231232376586907
Best medae: 0.9900000002235174
Best learning rate" 0.0019285697489976883

NeuralNet_logcosh
Best model params: {'activations': ('hard_sigmoid', 'sigmoid', 'sigmoid'), 'batch': 1230, 'loss': 'logcosh', 'nodes': (48, 4, 196), 'num_layers': 3, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001287)}
Best avg loss: 0.49966728687286377
Best avg Poisson deviance: 1.2619755560911738
Best avg RMSE: 1.2373130328035271
Best mae: 0.9489833746011163
Best avg mape: 1455189371021177.2
Best avg msle: 0.27701163774674004
Best avg r2: -0.016567145838368935
Best medae: 0.8020133376121521
Best learning rate" 0.0012877876870334148




NO TRANSFORMATION
CatBoost_mae
Best model params: {'bagging_temperature': 69.32347629336867, 'colsample_bylevel': 0.9042263919423306, 'learning_rate': 0.059135216445649785, 'max_depth': 6, 'objective': 'MAE', 'random_strength': 57.5044030038614}
Best avg loss: 0.870191801817903
Best avg Poisson deviance: 1.1595571486227314
Best avg RMSE: 1.1729592987180506
Best mae: 0.870191801817903
Best avg mape: 1285043284909062.0
Best avg msle: 0.24907335306405007
Best avg r2: 0.08636610819820872
Best medae: 0.9649462519649902

NeuralNet_mae
Best model params: {'activations': ('hard_sigmoid', 'selu', 'tanh'), 'batch': 60, 'loss': 'mae', 'nodes': (14, 44, 104), 'num_layers': 3, 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0010012101847678423)}
Best avg loss: 0.860203309725246
Best avg Poisson deviance: 1.1826411371382328
Best avg RMSE: 1.1838243567694227
Best mae: 0.860203309725246
Best avg mape: 1327063779522472.5
Best avg msle: 0.2569999918484786
Best avg r2: 0.06932140358654235
Best medae: 0.9979725480079651
Best learning rate" 0.0010012101847678423

XGBoost_poisson
Best model params: {'colsample_bytree': 0.9, 'eta': 0.2, 'gamma': 1.25, 'max_depth': 3, 'min_child_weight': 6, 'n_estimators': 885, 'objective': 'count:poisson', 'subsample': 0.8}
Best avg loss: 0.8872984647750854
Best avg Poisson deviance: 1.1377375692284581
Best avg RMSE: 1.1641485920269763
Best mae: 0.8884959173898346
Best avg mape: 1318069609985741.8
Best avg msle: 0.24786355063027496
Best avg r2: 0.10004554525355999
Best medae: 0.8451168090105057

CatBoost_poisson
Best model params: {'bagging_temperature': 10.45943130929077, 'colsample_bylevel': 0.4316079607705634, 'learning_rate': 0.1811321733393647, 'max_depth': 4, 'objective': 'Poisson', 'random_strength': 38.80202988079048}
Best avg loss: 0.8756738901138306
Best avg Poisson deviance: 1.114488303567074
Best avg RMSE: 1.1483764115493074
Best mae: 0.9008554196253915
Best avg mape: 1470443815709423.8
Best avg msle: 0.25792567793283294
Best avg r2: 0.1241870460923902
Best medae: 0.8207885864489404

NeuralNet_poisson
Best model params: {'activations': ('sigmoid', 'sigmoid'), 'batch': 994, 'loss': 'poisson', 'nodes': (2, 306), 'num_layers': 2, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001287)}
Best avg loss: 0.9405314326286316
Best avg Poisson deviance: 1.224157566651796
Best avg RMSE: 1.2169772920312838
Best mae: 0.9776571804501608
Best avg mape: 1644342233625140.0
Best avg msle: 0.2873375897024306
Best avg r2: 0.016554296086070203
Best medae: 0.6578687131404877

HistGradientBoost
Best model params: {'learning_rate': 0.10109485695245424, 'max_leaf_nodes': 3, 'min_samples_leaf': 57}
Best avg loss: 0.8755381405353546
Best avg Poisson deviance: 1.1142167902683617
Best avg RMSE: 1.1478974003153342
Best mae: 0.9007750488665334
Best avg mape: 1472622139555473.8
Best avg msle: 0.25796459486529216
Best avg r2: 0.12494601303114578
Best medae: 0.8181261976644996

XGBoost_sqrd_err
Best model params: {'colsample_bytree': 0.75, 'eta': 0.25, 'gamma': 1.4000000000000001, 'max_depth': 3, 'min_child_weight': 4, 'n_estimators': 291, 'objective': 'reg:squarederror', 'subsample': 0.8500000000000001}
Best avg loss: 1.3538366352907696
Best avg Poisson deviance: 1.1382166069014454
Best avg RMSE: 1.1635377148526194
Best mae: 0.8874297463584777
Best avg mape: 1317188436837995.5
Best avg msle: 0.24797228536579546
Best avg r2: 0.10097482606051361
Best medae: 0.8544332385063171


NeuralNet_mse
Best model params: {'activations': ('sigmoid', 'tanh'), 'batch': 308, 'loss': 'mse', 'nodes': (52, 18), 'num_layers': 2, 'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001)}
Best avg loss: 1.3119802751235328
Best avg Poisson deviance: 1.1106765851339446
Best avg RMSE: 1.1454141812578134
Best mae: 0.8997137295751303
Best avg mape: 1472651068455487.5
Best avg msle: 0.2575777145022419
Best avg r2: 0.12869477801471202
Best medae: 0.820500023663044

ExtraTrees
Best model params: {'max_features': 'log2', 'min_samples_leaf': 7, 'min_samples_split': 7, 'n_estimators': 478}
Best avg loss: 1.318912282274133
Best avg Poisson deviance: 1.1151041652936946
Best avg RMSE: 1.1484374809825924
Best mae: 0.9027866844755217
Best avg mape: 1483469242147171.2
Best avg msle: 0.2585737503010774
Best avg r2: 0.12412554898051564
Best medae: 0.8132442438836115

CatBoost_huber
Best model params: {'bagging_temperature': 8.06887605641524, 'colsample_bylevel': 0.4750844452806466, 'learning_rate': 0.08477793380600201, 'max_depth': 4, 'objective': 'Huber:delta=1', 'random_strength': 91.32730657879162}
Best avg loss: 0.5114836692810059
Best avg Poisson deviance: 1.128186222680005
Best avg RMSE: 1.1554092005771577
Best mae: 0.8903007943333388
Best avg mape: 1320537987430003.0
Best avg msle: 0.24807260796840236
Best avg r2: 0.11345196732275617
Best medae: 0.8133270033386434

NeuralNet_huber
Best model params: {'activations': ('sigmoid', 'hard_sigmoid', 'sigmoid'), 'batch': 1002, 'loss': 'huber_loss', 'nodes': (4, 446, 20), 'num_layers': 3, 'optimizer': tf.keras.optimizers.SGD(learning_rate=0.001046389}
Best avg loss: 0.5632255673408508
Best avg Poisson deviance: 1.2596914802385706
Best avg RMSE: 1.2361411962423543
Best mae: 0.9523753711625803
Best avg mape: 1472440814117645.0
Best avg msle: 0.27807662934368754
Best avg r2: -0.014615806608657744
Best medae: 0.7880968749523163
Best learning rate" 0.0010463893413543701

'''
