import pandas as pd
import numpy as np

import tensorflow as tf
from data_operations import Bookmaker
from analysis import Distribution
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


df['spi_diff'] = df['spi1'] - df['spi2']
df['convrate_diff'] = df['convrate1_home'] + df['convrate2_away']
df['avg_xg_diff'] = df['avg_xg1_home'] + df['avg_xg2_away']
df['avg_xg_diff_home'] = df['avg_xg1_home'] - df['avg_xg2_home']
df['avg_xg_diff_away'] = df['avg_xg1_away'] - df['avg_xg2_away']

test_matches_number = 3000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()



# features = ['adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home', 'xdraw_home',
# 			'xwin2_home', 'shots1_home', 'shots2_home','shotsot1_home', 'shotsot2_home',
# 			'corners1_home', 'corners2_home', 'xpts1_home', 'xpts2_home', 'xgshot1_home',
# 			'convrate1_home',
# 			'adj_avg_xg1_away','adj_avg_xg2_away', 'xwin1_away', 'xdraw_away',
# 			'xwin2_away', 'shots1_away', 'shots2_away','shotsot1_away', 'shotsot2_away',
# 			'corners1_away', 'corners2_away', 'xpts1_away', 'xpts2_away',
# 			'A','H','score1_similar','score2_similar',
# 			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away',
# 			'score1', 'score2',

# 			'FTR', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
# 			'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5']

features_more = ['adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home', 'xdraw_home',
			'xwin2_home', 'shots1_home', 'shots2_home','shotsot1_home', 'shotsot2_home',
			'corners1_home', 'corners2_home', 'xpts1_home', 'xpts2_home', 'xgshot1_home',
			'convrate1_home',
			'adj_avg_xg1_away','adj_avg_xg2_away', 'xwin1_away', 'xdraw_away',
			'xwin2_away', 'shots1_away', 'shots2_away','shotsot1_away', 'shotsot2_away',
			'corners1_away', 'corners2_away', 'xpts1_away', 'xpts2_away',
			'A','H','score1_similar','score2_similar',
			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away']

features_less = ['adj_avg_xg1_home','xwin1_home',
			'xwin2_home', 'shots1_home','shotsot1_home',
			'xpts1_home', 'xpts2_home', 
			'adj_avg_xg1_away', 'xwin1_away',
			'xwin2_away', 'shots1_away', 'shotsot1_away', 
			'xpts1_away', 'xpts2_away',
			'A','H','score1_similar','score2_similar',
			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away']

strght_features_more = ['strght_adj_avg_xg1_home','strght_adj_avg_xg2_home','strght_xwin1_home', 'strght_xdraw_home',
			'strght_xwin2_home', 'strght_shots1_home', 'strght_shots2_home','strght_shotsot1_home', 'strght_shotsot2_home',
			'strght_corners1_home', 'strght_corners2_home', 'strght_xpts1_home', 'strght_xpts2_home', 'strght_xgshot1_home',
			'strght_convrate1_home',
			'strght_adj_avg_xg1_away','strght_adj_avg_xg2_away', 'strght_xwin1_away', 'strght_xdraw_away',
			'strght_xwin2_away', 'strght_shots1_away', 'strght_shots2_away','strght_shotsot1_away', 'strght_shotsot2_away',
			'strght_corners1_away', 'strght_corners2_away', 'strght_xpts1_away', 'strght_xpts2_away',
			'strght_A','strght_H','strght_score1_similar','strght_score2_similar',
			'strght_spi_diff', 'strght_avg_xg_diff', 'strght_avg_xg_diff_home', 'strght_avg_xg_diff_away']

strght_features_less = ['strght_adj_avg_xg1_home','strght_xwin1_home',
			'strght_xwin2_home', 'strght_shots1_home','strght_shotsot1_home',
			'strght_xpts1_home', 'strght_xpts2_home', 
			'strght_adj_avg_xg1_away', 'strght_xwin1_away',
			'strght_xwin2_away', 'strght_shots1_away', 'strght_shotsot1_away', 
			'strght_xpts1_away', 'strght_xpts2_away',
			'strght_A','strght_H','strght_score1_similar','strght_score2_similar',
			'strght_spi_diff', 'strght_avg_xg_diff', 'strght_avg_xg_diff_home', 'strght_avg_xg_diff_away']


dist = Distribution()
for column in df_train:
	if df_train[column].dtype == 'float64':
		if not df_train[column].lt(0).any():
			df_train[f'strght_{column}'] = dist.straighten(df_train[column] + 0.001)
		else:
			df_train[f'strght_{column}'] = df_train[column]

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

# neuralnet_space = {'batch_2': scope.int(hp.quniform('batch_2', 32, 2048, 2)),		
# 								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
# 								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
# 								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
# 								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
# 								'loss_2': hp.choice('loss_2', ['mae', 'logcosh', 'huber_loss']),
# 								'optimizer_2': hp.choice('optimizer_2', [
# 									scope.Adam(
# 										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.005))),
# 									scope.Adagrad(
# 										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.005))),
# 									scope.RMSprop(
# 										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.005))),
# 									scope.Nadam(
# 										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.005))),
# 									scope.SGD(
# 										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.005)))
# 									])}

neuralnet_space = hp.choice('model',
			[{'num_layers': 2,  'batch_2': scope.int(hp.quniform('batch_2', 32, 2048, 2)),		
								'activations_21': hp.choice('activations_21', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_22': hp.choice('activations_22', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_21': scope.int(hp.qloguniform('nodes_21', np.log(2), np.log(1024), 2)),
								'nodes_22': scope.int(hp.qloguniform('nodes_22', np.log(2), np.log(1024), 2)),
								'loss_2': hp.choice('loss_2', ['poisson', 'mae', 'logcosh', 'huber_loss']),
								'features': hp.choice('features_2', [features_more, features_less, strght_features_more, strght_features_less]),
								'optimizer_2': hp.choice('optimizer_2', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_2', np.log(0.001), np.log(0.005))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_2', np.log(0.001), np.log(0.005))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_2', np.log(0.001), np.log(0.005))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_2', np.log(0.001), np.log(0.005))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_2', np.log(0.001), np.log(0.005)))
									])},
			 {'num_layers': 3,  'batch_3': scope.int(hp.quniform('batch_3', 32, 2048, 2)),		
								'activations_31': hp.choice('activations_31', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_32': hp.choice('activations_32', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'activations_33': hp.choice('activations_33', ['relu', 'tanh', 'sigmoid', 'selu', 'hard_sigmoid']),
								'nodes_31': scope.int(hp.qloguniform('nodes_31', np.log(2), np.log(1024), 2)),
								'nodes_32': scope.int(hp.qloguniform('nodes_32', np.log(2), np.log(1024), 2)),
								'nodes_33': scope.int(hp.qloguniform('nodes_33', np.log(2), np.log(1024), 2)),
								'loss_3': hp.choice('loss_3', ['poisson', 'mae', 'logcosh', 'huber_loss']),
								'features': hp.choice('features_3', [features_more, features_less, strght_features_more, strght_features_less]),
								'optimizer_3': hp.choice('optimizer_3', [
									scope.Adam(
										learning_rate=hp.loguniform('adam_learning_rate_3', np.log(0.001), np.log(0.005))),
									scope.Adagrad(
										learning_rate=hp.loguniform('adagrad_learning_rate_3', np.log(0.001), np.log(0.005))),
									scope.RMSprop(
										learning_rate=hp.loguniform('rmsprop_learning_rate_3', np.log(0.001), np.log(0.005))),
									scope.Nadam(
										learning_rate=hp.loguniform('nadam_learning_rate_3', np.log(0.001), np.log(0.005))),
									scope.SGD(
										learning_rate=hp.loguniform('sgd_learning_rate_3', np.log(0.001), np.log(0.005)))
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
		'features': hp.choice('features', [features_more, features_less, strght_features_more, strght_features_less]),
		'objective': hp.choice('objective', ['count:poisson', 'reg:squarederror'])
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
		'objective': hp.choice('objective', ['Poisson', 'MAE', 'Huber:delta=200']),
		'features': hp.choice('features', [features_more, features_less, strght_features_more, strght_features_less])
		#'pca_n_components': scope.int(hp.quniform('pca_n_components', 2, 30, 1))
		}

extratrees_space = {
		'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 1)),
		'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
		'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
		'features': hp.choice('features', [features_more, features_less, strght_features_more, strght_features_less])}

histgradboost_space = {
		'learning_rate': hp.uniform('learning_rate', 0.01, 0.8),
		'max_leaf_nodes': scope.int(hp.quniform('max_leaf_nodes', 2, 60, 1)),
		'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 60, 1)),
		'features': hp.choice('features', [features_more, features_less, strght_features_more, strght_features_less])}


models = {'XGBoost': xgb_space,
			'CatBoost': catboost_space,
			'NeuralNet': neuralnet_space,
			'HistGradientBoost': histgradboost_space
			#'ExtraTrees': extratrees_space,		
			}


res_dict = {}
def optimize(params, name, df_train, df_target):
	kf = KFold(n_splits=4)
	df_train_copy = df_train.copy()
	target_copy = df_target.copy()

	mae = []
	mse = []
	pdev = []
	rmse = []
	returns = []
	returns_over = []

	for train, test in kf.split(df_train_copy):
		X_train = df_train_copy[list(params['features'])].iloc[train]
		X_val = df_train_copy[list(params['features'])].iloc[test]
		y_train = target_copy.iloc[train]
		y_val = target_copy.iloc[test]

		y_train = np.sqrt(y_train+1)
		y_val = np.sqrt(y_val+1)

		score1_train = y_train.score1.values
		score1_val = y_val.score1.values
		score2_train = y_train.score2.values
		score2_val = y_val.score2.values

		scaler = StandardScaler()
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

			y_pred = np.power(model.predict(X_val), 2)-1
			score1_pred = y_pred[:, 0]
			score2_pred = y_pred[:, 1]
			score1_pred = score1_pred.clip(min=0.01)
			score2_pred = score2_pred.clip(min=0.01)
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
											min_samples_split=params['min_samples_split'],
											min_samples_leaf=params['min_samples_leaf'],
											max_features=params['max_features'])
				model.fit(X_train, score1_train)
			score1_pred = np.power(model.predict(X_val), 2)-1
			score1_pred = score1_pred.clip(min = .01)


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
											min_samples_split=params['min_samples_split'],
											min_samples_leaf=params['min_samples_leaf'],
											max_features=params['max_features'])
				model.fit(X_train, score2_train)


			score2_pred = np.power(model.predict(X_val), 2)-1
			score2_pred = score2_pred.clip(min = .01)

		foot_poisson = FootballPoissonModel()
		home_win, draw, away_win = foot_poisson.predict_chances(score1_pred, score2_pred)
		over, under = foot_poisson.predict_overs(score1_pred, score2_pred)
		df_predictions = df_train.iloc[test].copy()

		predictions = pd.DataFrame(data={'score1_pred': score1_pred, 'score2_pred': score2_pred,
										 'homewin_pred': np.clip(list(home_win), a_min=0.01, a_max=None), 'draw_pred': np.clip(list(draw), a_min=0.01, a_max=None), 'awaywin_pred': np.clip(list(away_win), a_min=0.01, a_max=None),
										 '>2.5_pred': over, '<2.5_pred': under})
		df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
		book = Bookmaker(df_predictions, odds='max', stake=5)
		book.calculate()

		score1_val = np.power(score1_val, 2)-1
		score2_val = np.power(score2_val, 2)-1

		mse.append((mean_squared_error(score1_val, score1_pred) + mean_squared_error(score2_val, score2_pred)) / 2)
		mae.append((mean_absolute_error(score1_val, score1_pred) + mean_absolute_error(score2_val, score2_pred)) / 2)
		pdev.append((mean_poisson_deviance(score1_val, score1_pred) + mean_poisson_deviance(score2_val, score2_pred)) / 2)
		rmse.append((np.sqrt(mean_squared_error(score1_val, score1_pred)) + np.sqrt(mean_squared_error(score2_val, score2_pred))) / 2)
		returns.append(df_predictions[['bet_return']].sum().values)
		returns_over.append(df_predictions[['bet_return_over']].sum().values)
	
	loss = np.mean(pdev)

	return {'status': 'ok',
			'loss': loss,
			'pdev': np.mean(pdev),
			'rmse': np.mean(rmse),
			'mae': np.mean(mae),
			'returns': np.mean(returns),
			'returns_over': np.mean(returns_over),
			'params': params,
			'model': model}


for model, space in models.items():
	print(model)
	opt_f = partial(optimize, name=model, df_train=df_train, df_target=target)
	trials = Trials()
	best = fmin(fn=opt_f,
	            space=space,
	            algo=tpe.suggest,
	            max_evals=100,
	            trials=trials)

	best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
	print(f'Best model params: {best_params}')
	best_loss = trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
	print(f'Best avg loss (p_dev): {best_loss}')
	best_pdev = trials.results[np.argmin([r['loss'] for r in trials.results])]['pdev']
	print(f'Best avg Poisson deviance: {best_pdev}')
	best_rmse = trials.results[np.argmin([r['loss'] for r in trials.results])]['rmse']
	print(f'Best avg RMSE: {best_rmse}')
	best_mae = trials.results[np.argmin([r['loss'] for r in trials.results])]['mae']
	print(f'Best mae: {best_mae}')
	best_returns = trials.results[np.argmin([r['loss'] for r in trials.results])]['returns']
	print(f'Best avg return (1X2 market): {best_returns}')
	best_returns_over = trials.results[np.argmin([r['loss'] for r in trials.results])]['returns_over']
	print(f'Best avg return (Over/Under 2.5 market): {best_returns_over}')
	best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
	if model == 'NeuralNet':
		print(f'Best learning rate" {best_model.model.optimizer.lr.numpy()}')

	res_dict[model] = [best_loss, best_pdev, best_rmse, best_mae, best_returns, best_returns_over]

results_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['loss', 'P DEVIANCE', 'RMSE', 'MAE', 'return_1X2', 'return_OU2.5'])
print(results_df.sort_values(by='MAE'))

'''
### XGBOOST
Best model params: {'colsample_bytree': 0.8500000000000001, 'eta': 0.35000000000000003, 'features': ('strght_adj_avg_xg1_home', 'strght_xwin1_home', 'strght_xwin2_home', 'strght_shots1_home', 'strght_shotsot1_home', 'strght_xpts1_home', 'strght_xpts2_home', 'strght_adj_avg_xg1_away', 'strght_xwin1_away', 'strght_xwin2_away', 'strght_shots1_away', 'strght_shotsot1_away', 'strght_xpts1_away', 'strght_xpts2_away', 'strght_A', 'strght_H', 'strght_score1_similar', 'strght_score2_similar', 'strght_spi_diff', 'strght_avg_xg_diff', 'strght_avg_xg_diff_home', 'strght_avg_xg_diff_away'), 'gamma': 2.4000000000000004, 'max_depth': 20, 'min_child_weight': 3, 'n_estimators': 342, 'objective': 'count:poisson', 'subsample': 0.7000000000000001}
Best avg loss (-sum_profit): -79.33766509404816
Best avg Poisson deviance: 0.23239267567267669
Best avg RMSE: 0.5386059408609108
Best mae: 0.43682466880748755
Best avg return (1X2 market): 79.33766509404816
Best avg return (Over/Under 2.5 market): -181.84713177917928

### CatBoost
Best model params: {'bagging_temperature': 36.377065381146764, 'colsample_bylevel': 0.558615051334185, 'features': ('strght_adj_avg_xg1_home', 'strght_xwin1_home', 'strght_xwin2_home', 'strght_shots1_home', 'strght_shotsot1_home', 'strght_xpts1_home', 'strght_xpts2_home', 'strght_adj_avg_xg1_away', 'strght_xwin1_away', 'strght_xwin2_away', 'strght_shots1_away', 'strght_shotsot1_away', 'strght_xpts1_away', 'strght_xpts2_away', 'strght_A', 'strght_H', 'strght_score1_similar', 'strght_score2_similar', 'strght_spi_diff', 'strght_avg_xg_diff', 'strght_avg_xg_diff_home', 'strght_avg_xg_diff_away'), 'learning_rate': 0.4318636879651188, 'max_depth': 7, 'objective': 'Huber:delta=200', 'random_strength': 82.23214945352323}
Best avg loss (-sum_profit): -82.41676329549992
Best avg Poisson deviance: 0.26293281511681954
Best avg RMSE: 0.5465776076810411
Best mae: 0.43912108084775203
Best avg return (1X2 market): 82.41676329549992
Best avg return (Over/Under 2.5 market): -124.28904693950872


'''

# xgb_params = {'colsample_bytree': 0.8500000000000001, 'eta': 0.35000000000000003, 
# 'features': ('strght_adj_avg_xg1_home', 'strght_xwin1_home', 'strght_xwin2_home', 'strght_shots1_home', 'strght_shotsot1_home', 'strght_xpts1_home', 'strght_xpts2_home', 
# 'strght_adj_avg_xg1_away', 'strght_xwin1_away', 'strght_xwin2_away', 'strght_shots1_away', 'strght_shotsot1_away', 'strght_xpts1_away', 'strght_xpts2_away', 'strght_A', 'strght_H', 'strght_score1_similar', 'strght_score2_similar', 'strght_spi_diff', 'strght_avg_xg_diff', 'strght_avg_xg_diff_home', 'strght_avg_xg_diff_away'), 
# 'gamma': 2.4000000000000004, 'max_depth': 20, 'min_child_weight': 3, 'n_estimators': 342, 'objective': 'count:poisson', 'subsample': 0.7000000000000001}


# df_train = df_train[list(xgb_params['features'])]

# X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.25, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.values)
# X_val = scaler.transform(X_val.values)
# X_test = scaler.transform(df_test[features_less].values)

# model_score1 = XGBRegressor(n_estimators=xgb_params['n_estimators'],
# 							eta=xgb_params['eta'],
# 							max_depth=xgb_params['max_depth'],
# 							min_child_weight=xgb_params['min_child_weight'],
# 							subsample=xgb_params['subsample'],
# 							gamma=xgb_params['gamma'],
# 							colsample_bytree=xgb_params['colsample_bytree'],
# 							#reg_lambda=params['lambda'],
# 							nthread=4,
# 							booster='gbtree',
# 							tree_method='exact')

# model_score1.fit(X_train, np.sqrt(y_train.score1.values+1), eval_metric="mae", eval_set=[(X_val, np.sqrt(y_val.score1.values+1))], early_stopping_rounds=10, verbose=True)

# model_score2 = XGBRegressor(n_estimators=xgb_params['n_estimators'],
# 							eta=xgb_params['eta'],
# 							max_depth=xgb_params['max_depth'],
# 							min_child_weight=xgb_params['min_child_weight'],
# 							subsample=xgb_params['subsample'],
# 							gamma=xgb_params['gamma'],
# 							colsample_bytree=xgb_params['colsample_bytree'],
# 							#reg_lambda=params['lambda'],
# 							nthread=4,
# 							booster='gbtree',
# 							tree_method='exact')

# model_score2.fit(X_train, np.sqrt(y_train.score2.values+1), eval_metric="mae", eval_set=[(X_val, np.sqrt(y_val.score2.values+1))], early_stopping_rounds=10, verbose=True)

# score1_pred = np.power(model_score1.predict(X_test), 2)-1
# score1_pred = score1_pred.clip(min = .01)

# score2_pred = np.power(model_score2.predict(X_test), 2)-1
# score2_pred = score2_pred.clip(min = .01)

# foot_poisson = FootballPoissonModel()
# home_win, draw, away_win = foot_poisson.predict_chances(score1_pred, score2_pred)
# over, under = foot_poisson.predict_overs(score1_pred, score2_pred)
# df_predictions = df_test.copy()

# predictions = pd.DataFrame(data={'score1_pred': score1_pred, 'score2_pred': score2_pred,
# 							 'homewin_pred': np.clip(list(home_win), a_min=0.01, a_max=None), 'draw_pred': np.clip(list(draw), a_min=0.01, a_max=None), 'awaywin_pred': np.clip(list(away_win), a_min=0.01, a_max=None),
# 							 '>2.5_pred': over, '<2.5_pred': under})
# df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
# book = Bookmaker(df_predictions, odds='max', stake=5)
# book.calculate()
# print(df_predictions.describe())

# print(df_predictions[['bet_return']].sum().values)
# print(df_predictions[['bet_return_over']].sum().values)