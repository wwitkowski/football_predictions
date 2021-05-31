import pandas as pd
import numpy as np

import tensorflow as tf
from data_operations import Bookmaker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from models import NeuralNetworkModel, FootballPoissonModel

from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
df = pd.read_csv('training_data\\training_data_decay00325_num40_wluck.csv')
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

print(df.shape)

df['spi_diff'] = df['spi1'] - df['spi2']
df['convrate_diff'] = df['convrate1_home'] + df['convrate2_away']
df['avg_xg_diff'] = df['avg_xg1_home'] + df['avg_xg2_away']
df['avg_xg_diff_home'] = df['avg_xg1_home'] - df['avg_xg2_home']
df['avg_xg_diff_away'] = df['avg_xg1_away'] - df['avg_xg2_away']

test_matches_number = 3000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()



features = ['adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home', 'xdraw_home',
			'xwin2_home', 'shots1_home', 'shots2_home','shotsot1_home', 'shotsot2_home',
			'corners1_home', 'corners2_home', 'xpts1_home', 'xpts2_home', 'xgshot1_home',
			'convrate1_home',
			'adj_avg_xg1_away','adj_avg_xg2_away', 'xwin1_away', 'xdraw_away',
			'xwin2_away', 'shots1_away', 'shots2_away','shotsot1_away', 'shotsot2_away',
			'corners1_away', 'corners2_away', 'xpts1_away', 'xpts2_away',
			'A','H','score1_similar','score2_similar',
			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away',
			'score1', 'score2',

			'FTR', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
			'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5']

train_features = ['adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home', 'xdraw_home',
			'xwin2_home', 'shots1_home', 'shots2_home','shotsot1_home', 'shotsot2_home',
			'corners1_home', 'corners2_home', 'xpts1_home', 'xpts2_home', 'xgshot1_home',
			'convrate1_home',
			'adj_avg_xg1_away','adj_avg_xg2_away', 'xwin1_away', 'xdraw_away',
			'xwin2_away', 'shots1_away', 'shots2_away','shotsot1_away', 'shotsot2_away',
			'corners1_away', 'corners2_away', 'xpts1_away', 'xpts2_away',
			'A','H','score1_similar','score2_similar',
			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away']

# features = ['adj_avg_xg1_home','xwin1_home',
# 			'xwin2_home', 'shots1_home','shotsot1_home',
# 			'xpts1_home', 'xpts2_home', 
# 			'adj_avg_xg1_away', 'xwin1_away',
# 			'xwin2_away', 'shots1_away', 'shotsot1_away', 
# 			'xpts1_away', 'xpts2_away',
# 			'A','H','score1_similar','score2_similar',
# 			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away']


df_train = df_train[features]
df_test = df_test[features]
target = df_train[['score1', 'score2']]
target_test = df_test[['score1', 'score2']]

print(f'Number of NA values: {df_train.isna().sum().sum()}')
print(f'Number of INF values: {df_train.replace([np.inf, -np.inf], np.nan).isna().sum().sum()}')

try:
	scope.define(tf.keras.optimizers.Adam)
	scope.define(tf.keras.optimizers.Adagrad)
	scope.define(tf.keras.optimizers.RMSprop)
	scope.define(tf.keras.optimizers.Nadam)
	scope.define(tf.keras.optimizers.SGD)
except ValueError:
	pass

neuralnet_space = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 2048, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['poisson', 'mae', 'logcosh', 'huber_loss']),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.1))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.1))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.1))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.1))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.1)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 2048, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['poisson', 'mae', 'logcosh', 'huber_loss']),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.1))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.1))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.1))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.1))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.1)))
									])},
			 {'num_layers': 4,  'batch_4': scope.int(hp.quniform('batch_4', 32, 2048, 2)),		
								'activations_41': hp.choice('activations_41', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_42': hp.choice('activations_42', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_43': hp.choice('activations_43', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_44': hp.choice('activations_44', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_41': scope.int(hp.qloguniform('nodes_41', np.log(2), np.log(1024), 2)),
								'nodes_42': scope.int(hp.qloguniform('nodes_42', np.log(2), np.log(1024), 2)),
								'nodes_43': scope.int(hp.qloguniform('nodes_43', np.log(2), np.log(1024), 2)),
								'nodes_44': scope.int(hp.qloguniform('nodes_44', np.log(2), np.log(1024), 2)),
								'loss_4': hp.choice('loss_4', ['poisson', 'mae', 'logcosh', 'huber_loss']),
								'optimizer_4': hp.choice('optimizer_4', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_4', np.log(0.001), np.log(0.1))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_4', np.log(0.001), np.log(0.1))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_4', np.log(0.001), np.log(0.1))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_4', np.log(0.001), np.log(0.1))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_4', np.log(0.001), np.log(0.1)))
									])}
								])

xgb_space = {
		'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
		'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
		'max_depth':  scope.int(hp.quniform('max_depth', 2, 20, 1)),
		'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1)),
		'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
		'gamma': hp.quniform('gamma', 0.5, 3, 0.05),
		'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
		'objective': hp.choice('objective', ['count:poisson', 'reg:squarederror', 'reg:pseudohubererror'])
		#'lambda': hp.quniform('lambda', 0, 5, 0.1),
		#'alpha': hp.quniform('alpha', 0, 3, 0.1),
		}

catboost_space = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		#'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.0, 5.0),
		'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
		'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1.0),
		'bagging_temperature': hp.uniform('bagging_temperature', 0.0, 100),
		'random_strength': hp.uniform('random_strength', 0.0, 100),
		'objective': hp.choice('objective', ['Poisson', 'MAE', 'Huber:delta=200'])
		#'pca_n_components': scope.int(hp.quniform('pca_n_components', 2, 30, 1))
		}

extratrees_space = {
		'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
		'max_depth': scope.int(hp.quniform('max_depth', 2, 20, 1)),
		'min_samples_split': scope.int(hp.quniform('min_samples_split', 1, 6, 1)),
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
		'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2'])}

histgradboost_space = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		'max_leaf_nodes': scope.int(hp.quniform('max_leaf_nodes', 2, 60, 1)),
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1))}


models = {'XGBoost': xgb_space,
			'HistGradientBoost': histgradboost_space,
			'ExtraTrees': extratrees_space,
			'CatBoost': catboost_space,
			'NeuralNet': neuralnet_space}


res_dict = {}
def optimize(params, name, df_train, df_test):
	kf = KFold(n_splits=6)
	df_train_copy = df_train.copy()
	target_copy = df_test.copy()

	mae = []
	mse = []
	pdev = []
	rmse = []
	returns = []
	returns_over = []

	for train, test in kf.split(df_train_copy):
		X_train = df_train_copy[train_features].iloc[train]
		X_val = df_train_copy[train_features].iloc[test]
		y_train = target_copy.iloc[train]
		y_val = target_copy.iloc[test]

		score1_train = y_train.score1.values
		score1_val = y_val.score1.values
		score2_train = y_train.score2.values
		score2_val = y_val.score2.values

		scaler = MinMaxScaler()
		X_train = scaler.fit_transform(X_train.values)
		X_val = scaler.transform(X_val.values)


		if 'NeuralNet' in name:
			model = NeuralNetworkModel()
			if params['num_layers'] == 2:
				model.build(n_features=X_train.shape[1],
							optimizer=params['optimizer_2'],
							#dropout=params['dropout_2'],
							activations=(params['activations_21'], params['activations_22']),
							nodes=(params['nodes_21'], params['nodes_22']),
							loss=params['loss_2'],
							metrics=['mse', 'mae'])

				history = model.train(X_train, y_train.values, X_val, y_val.values,
								   		verbose=0,
								   		batch_size=params['batch_2'], 
								   		epochs=500)

			elif params['num_layers'] == 3:
				model.build(n_features=X_train.shape[1],
							optimizer=params['optimizer_3'],
							#dropout=params['dropout_3'],
							activations=(params['activations_31'], params['activations_32'], params['activations_33']),
							nodes=(params['nodes_31'], params['nodes_32'], params['nodes_33']),
							loss=params['loss_3'],
							metrics=['mse', 'mae'])

				history = model.train(X_train, y_train.values, X_val, y_val.values,
								   		verbose=0,
								   		batch_size=params['batch_3'], 
								   		epochs=500)

			else:
				model.build(n_features=X_train.shape[1],
							optimizer=params['optimizer_4'],
							#dropout=params['dropout_4'],
							activations=(params['activations_41'], params['activations_42'], params['activations_43'], params['activations_44']),
							nodes=(params['nodes_41'], params['nodes_42'], params['nodes_43'], params['nodes_44']),
							loss=params['loss_4'],
							metrics=['mse', 'mae'])

				history = model.train(X_train, y_train.values, X_val, y_val.values,
								   		verbose=0,
								   		batch_size=params['batch_4'], 
								   		epochs=500)

			y_pred = model.predict(X_val)
			score1_pred = y_pred[:, 0]
			score2_pred = y_pred[:, 1]
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
				model.fit(X_train, score1_train, eval_set=(X_val, score1_val), early_stopping_rounds=10)

			elif 'XGBoost' in name:
				model = XGBRegressor(n_estimators=params['n_estimators'],
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
				model.fit(X_train, score1_train, eval_metric="mae", eval_set=[(X_val, score1_val)], early_stopping_rounds=10, verbose=False)
			elif 'HistGradientBoost' in name:
				model = HistGradientBoostingRegressor(loss='poisson',
													learning_rate=params['learning_rate'],
													max_leaf_nodes=params['max_leaf_nodes'],
													min_samples_leaf=params['min_samples_leaf'])
				model.fit(X_train, score1_train)
			else:
				model = ExtraTreesRegressor(n_estimators=params['n_estimators'],
											max_depth=params['max_depth'],
											min_samples_split=params['min_samples_split'],
											min_samples_leaf=params['min_samples_leaf'],
											max_features=params['max_features'])
				model.fit(X_train, score1_train)
			score1_pred = model.predict(X_val)
			score1_pred = score1_pred.clip(min = .0001)


			if 'CatBoost' in name:
				model = CatBoostRegressor(objective=params['objective'],
									learning_rate=params['learning_rate'],
									#l2_leaf_reg=score1_params['l2_leaf_reg'],
									max_depth=params['max_depth'],
									colsample_bylevel=params['colsample_bylevel'],
									bagging_temperature=params['bagging_temperature'],
									random_strength=params['random_strength'],
									verbose=0)
				model.fit(X_train, score2_train, eval_set=(X_val, score2_val), early_stopping_rounds=10)

			elif 'XGBoost' in name:
				model = XGBRegressor(n_estimators=params['n_estimators'],
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
				model.fit(X_train, score2_train, eval_metric="mae", eval_set=[(X_val, score2_val)], early_stopping_rounds=10, verbose=False)
			elif 'HistGradientBoost' in name:
				model = HistGradientBoostingRegressor(loss='poisson',
													learning_rate=params['learning_rate'],
													max_leaf_nodes=params['max_leaf_nodes'],
													min_samples_leaf=params['min_samples_leaf'])
				model.fit(X_train, score2_train)
			else:
				model = ExtraTreesRegressor(n_estimators=params['n_estimators'],
											max_depth=params['max_depth'],
											min_samples_split=params['min_samples_split'],
											min_samples_leaf=params['min_samples_leaf'],
											max_features=params['max_features'])
				model.fit(X_train, score2_train)


			score2_pred = model.predict(X_val)
			score2_pred = score2_pred.clip(min = .0001)

		foot_poisson = FootballPoissonModel()
		home_win, draw, away_win = foot_poisson.predict_chances(score1_pred, score2_pred)
		over, under = foot_poisson.predict_overs(score1_pred, score2_pred)
		df_predictions = df_train.iloc[test].copy()

		predictions = pd.DataFrame(data={'score1_pred': score1_pred, 'score2_pred': score2_pred,
										 'homewin_pred': home_win, 'draw_pred': draw, 'awaywin_pred': away_win,
										 '>2.5_pred': over, '<2.5_pred': under})
		df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
		book = Bookmaker(df_predictions, odds='max', stake=5)
		book.calculate()

		mse.append((mean_squared_error(score1_val, score1_pred) + mean_squared_error(score2_val, score2_pred)) / 2)
		mae.append((mean_absolute_error(score1_val, score1_pred) + mean_absolute_error(score2_val, score2_pred)) / 2)
		pdev.append((mean_poisson_deviance(score1_val, score1_pred) + mean_poisson_deviance(score2_val, score2_pred)) / 2)
		rmse.append((np.sqrt(mean_squared_error(score1_val, score1_pred)) + np.sqrt(mean_squared_error(score2_val, score2_pred))) / 2)
		returns.append(df_predictions[['bet_return']].sum().values)
		returns_over.append(df_predictions[['bet_return_over']].sum().values)
	
	loss = np.mean(mae)

	return {'status': 'ok',
			'loss': loss,
			'pdev': np.mean(pdev),
			'rmse': np.mean(rmse),
			'returns': np.mean(returns),
			'returns_over': np.mean(returns_over),
			'params': params,
			'model': model}


for model, space in models.items():
	print(model)
	opt_f = partial(optimize, name=model, df_train=df_train, df_test=target)
	trials = Trials()
	best = fmin(fn=opt_f,
	            space=space,
	            algo=tpe.suggest,
	            max_evals=50,
	            trials=trials)

	best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
	print(f'Best model params: {best_params}')
	best_loss = trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
	print(f'Best avg loss (MAE): {best_loss}')
	best_pdev = trials.results[np.argmin([r['loss'] for r in trials.results])]['pdev']
	print(f'Best avg Poisson deviance: {best_pdev}')
	best_rmse = trials.results[np.argmin([r['loss'] for r in trials.results])]['rmse']
	print(f'Best avg RMSE: {best_rmse}')
	best_returns = trials.results[np.argmin([r['loss'] for r in trials.results])]['returns']
	print(f'Best avg return (1X2 market): {best_returns}')
	best_returns_over = trials.results[np.argmin([r['loss'] for r in trials.results])]['returns_over']
	print(f'Best avg return (Over/Under 2.5 market): {best_returns_over}')
	best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
	if model == 'NeuralNet':
		print(f'Best learning rate" {best_model.model.optimizer.lr.numpy()}')

	res_dict[model] = [best_loss, best_pdev, best_rmse, best_returns, best_returns_over]

results_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['MAE', 'P DEVIANCE', 'RMSE', 'return_1X2', 'return_OU2.5'])
print(results_df.sort_values(by='MAE'))
