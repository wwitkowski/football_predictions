from data_operations import FootballStats
from data_operations import TeamStats
from data_operations import Bookmaker
from datetime import datetime, timedelta
from analysis import Distribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from models import NeuralNetworkModel, FootballPoissonModel

from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

DATE = datetime.now().strftime('%Y-%m-%d')
EXCLUDE_FEATURES = ['prob1', 'prob2', 'probtie', 
					'proj_score1', 'proj_score2', 'MaxH', 'MaxD', 'MaxA',
					'AvgH', 'AvgD', 'AvgA', 'league_id', 'season']

ts = TeamStats(decay_factor=0.00325, num_of_matches=40, min_matches=8)
final_data = pd.DataFrame()
counter = 0

with FootballStats() as stats:
	for date in stats.data.date.unique():
		if date < '2017-01-01':
			continue
		elif date == DATE:
			break
		try:
			date_data = ts.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
			final_data = pd.concat([final_data, date_data], sort=False, ignore_index=True)
		except ValueError as e:
			pass
		counter += 1
		print('Calculating stats... {0:.2f}% done'.format((counter / len(stats.data.date.unique())) * 100), end="\r")

final_data.to_csv('training_data\\training_data_decay00325_num40_wluck.csv', index=False)


df = pd.read_csv('training_data\\training_data_decay00325_num40_wluck.csv')
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

test_matches_number = 2000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()

target = df_train[['score1', 'score2']]
df_train.drop(['score1', 'score2'], axis=1, inplace=True)
print(f'Number of NA values: {df_train.isna().sum().sum()}')
print(f'Number of INF values: {df_train.replace([np.inf, -np.inf], np.nan).isna().sum().sum()}')

df_train['spi_diff'] = df_train['spi1'] - df_train['spi2']
features = ['avg_xg1_home', 'xgshot1_home', 'shots1_home', 'shotsot1_home', 'shotsot2_home', 
		'shots2_home', 'corners1_home', 'corners2_home', 'fouls1_home', 'xpts1_home', 'convrate1_home',
		'avg_xg1_away','shots1_away', 'shotsot1_away', 'shots2_away', 'shotsot2_away', 'corners1_away',
		'corners2_away', 'xpts1_away', 'convrate1_away', 'importance1', 'importance2', 'xg1_similar', 
		'xg2_similar', 'D', 'spi_diff']

df_train = df_train[features]
sns.pairplot(df_train)
plt.show()

df_train_strght = pd.DataFrame()
dist = Distribution()
for column in df_train:
	if df_train[column].dtype == 'float64':
		df_train_strght[f'strght_{column}'] = dist.straighten(df_train[column] + 0.001)

sns.pairplot(df_train_strght)
plt.show()


df_variance = df_train.var()
print(df_variance)


# # goals_pos_luck = df_train[['xg1']].loc[df_train['importance_diff'] > 30]
# # goals_neg_luck = df_train[['xg1']].loc[df_train['importance_diff'] <= 30]
# # means = (goals_pos_luck.mean().values, goals_neg_luck.mean().values)
# # print(goals_pos_luck.describe())
# # print(goals_neg_luck.describe())

# # df_train2 = df_train[['xg1', 'xg2', 'shotsot1_home', 'shotsot2_away', 'FTR']]
# # sns.pairplot(df_train2, hue='FTR')
# # plt.show()




# df_train = df_train[score1_features]



# X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.1, random_state=0)


# # Scale data
# scaler = StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
# X_val = pd.DataFrame(scaler.transform(X_val.values), columns=X_val.columns, index=X_val.index)





# # for column in X_train:
# # 	p_corr, _ = pearsonr(X_train[column], y_train)
# # 	sp_corr, _ = spearmanr(X_train[column], y_train)
# # 	print(f'Pearson corr {column}: {p_corr}')
# # 	print(f'Spearman corr {column}: {sp_corr}')

# # corr_matrx = X_train[features].corr()
# # sns.heatmap(corr_matrx, annot=True, cmap="YlGnBu")
# # plt.show()

# # pca = PCA(n_components=2)
# # X_train_pca = pca.fit_transform(X_train)
# # X_val_pca = pca.transform(X_val)
# # plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5,
# #         align='center', label='individual explained variance')

# # plt.show()
# # print(sum(pca.explained_variance_ratio_))



# # alpha = np.linspace(0.001, 0.01, 101)
# # print(alpha)
# # search = GridSearchCV(estimator=Lasso(), param_grid={'alpha': alpha}, cv=5, scoring='neg_mean_absolute_error', verbose=3)
# # search.fit(X_train, y_train)

# # lasso = Lasso(alpha=0.001)
# # lasso.fit(X_train, y_train)

# # lasso_y_pred = lasso.predict(X_val)
# # print(f"MAE: {mean_absolute_error(y_val, lasso_y_pred, multioutput='raw_values')}")
# # print(f"MSE: {mean_squared_error(y_val, lasso_y_pred, multioutput='raw_values')}")


# # print('xg_boost')
# # xgb_model_score1 = xgb.XGBRegressor()
# # xgb_model_score1.fit(X_train, y_train.score1)
# # xgb_model_score2 = xgb.XGBRegressor()
# # xgb_model_score2.fit(X_train, y_train.score2)

# # xgb_score1_pred = xgb_model_score1.predict(X_val)
# # xgb_score2_pred = xgb_model_score2.predict(X_val)

# # print(f"MAE score1: {mean_absolute_error(y_val.score1, xgb_score1_pred)}")
# # print(f"MAE score2: {mean_absolute_error(y_val.score2, xgb_score2_pred)}")

# # print('cat_boost')
# # cb_model_score1 = CatBoostRegressor()
# # cb_model_score1.fit(X_train, y_train.score1, eval_set=(X_val, y_val.score1), early_stopping_rounds=10)
# # cb_model_score2 = CatBoostRegressor()
# # cb_model_score2.fit(X_train, y_train.score2, eval_set=(X_val, y_val.score2), early_stopping_rounds=10)

# # cb_score1_pred = cb_model_score1.predict(X_val)
# # cb_score2_pred = cb_model_score2.predict(X_val)

# # print(f"MAE score1: {mean_absolute_error(y_val.score1, cb_score1_pred)}")
# # print(f"MAE score2: {mean_absolute_error(y_val.score2, cb_score2_pred)}")

# print('neural network')


# activations = ('sigmoid', 'tanh')
# nodes = (120, 118)
# nn_model = NeuralNetworkModel()
# nn_model.build(n_features=X_train.shape[1],
# 				optimizer='nadam',
# 				dropout=0.45,
# 				activations=activations,
# 				nodes=nodes)

# print(nn_model.summary)

# # history = nn_model.train(X_train.values, y_train.values, X_val.values, y_val.values, 
# # 						verbose=0, batch_size=782, epochs=500)


# # best_model = NeuralNetworkModel('nn_model')

# # df_test['spi_diff'] = df_test['spi1'] - df_test['spi2']
# # X_test = df_test[score1_features]
# # X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns, index=X_test.index)
# # y_test = df_test[['score1', 'score2']]
# # nn_y_pred = best_model.predict(X_test.values)

# # pred_home_goals = nn_y_pred[:, 0]
# # pred_away_goals = nn_y_pred[:, 1]
# # foot_poisson = FootballPoissonModel()
# # home_win, draw, away_win = foot_poisson.predict_chances(pred_home_goals, pred_away_goals)

# # df_predictions = df_test[['date', 'league', 'team1', 'team2', 'score1', 'score2', 'FTR', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 
# # 						  'shots1', 'shots2', 'shotsot1', 'shotsot2', 'fouls1', 'fouls2', 'corners1', 'corners2', 
# # 						  'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'xg1_similar', 'xg2_similar', 'H', 'D', 'A']]

# # predictions = pd.DataFrame(data={'score1_pred': pred_home_goals, 'score2_pred': pred_away_goals,
# # 								 'homewin_pred': home_win, 'draw_pred': draw, 'awaywin_pred': away_win})


# # df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
# # #print(df_predictions.FTR)

# # book = Bookmaker(df_predictions, stake=5)
# # book.calculate()
# # #print(df_predictions[['FTR', 'BET', 'prediction_odds', 'bet_return']].head(20))
# # print(df_predictions[['bet_return']].sum().values)








# # print(f"MAE: {mean_absolute_error(y_val, nn_y_pred, multioutput='raw_values')}")
# # print(f"MSE: {mean_squared_error(y_val, nn_y_pred, multioutput='raw_values')}")


# # print('Neural network KFOOLD CV')

# # space = {
# # 		'batch': scope.int(hp.quniform('batch', 32, 2048, 2)),
# # 		'activations_1': hp.choice('activations_1', ['relu', 'tanh', 'sigmoid']),
# # 		'activations_2': hp.choice('activations_2', ['relu', 'tanh', 'sigmoid']),
# # 		'activations_3': hp.choice('activations_3', ['relu', 'tanh', 'sigmoid']),
# # 		'nodes_1': scope.int(hp.qloguniform('nodes_1', np.log(2), np.log(512), 2)),
# # 		'nodes_2': scope.int(hp.qloguniform('nodes_2', np.log(2), np.log(512), 2)),
# # 		'nodes_3': scope.int(hp.qloguniform('nodes_3', np.log(2), np.log(512), 2)),
# # 		'dropout': hp.uniform('dropout', 0.01, 0.5),
# # 		'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'sgd', 'rmsprop'])
# # 		}


# # def optimize(params, df_train, df_test):

# # 	kf = KFold(n_splits=8)
# # 	df_train_copy = df_train.copy()
# # 	target_copy = df_test.copy()
# # 	mae = []
# # 	mse = []
# # 	for train, test in kf.split(df_train_copy):
# # 		scaler = StandardScaler()
# # 		X_train = scaler.fit_transform(df_train_copy.iloc[train])
# # 		X_val = scaler.transform(df_train_copy.iloc[test])

# # 		y_train = target_copy.iloc[train].values
# # 		y_val = target_copy.iloc[test].values

# # 		nn_model = NeuralNetworkModel()
# # 		nn_model.build(n_features=X_train.shape[1],
# # 						optimizer=params['optimizer'],
# # 						dropout=params['dropout'],
# # 						activations=(params['activations_1'], params['activations_2'], params['activations_3']),
# # 						nodes=(params['nodes_1'], params['nodes_2'], params['nodes_3']))

# # 		history = nn_model.train(X_train, y_train, X_val, y_val,
# # 							   			 verbose=0,
# # 							   			 batch_size=params['batch'], 
# # 							   			 epochs=500)

# # 		mse.append(np.min(history.history['val_loss']))
# # 		mae.append(np.min(history.history['val_mae']))
# # 	#print(f'Average min loss: {np.mean(mae)}, Std dev: {np.std(mae)}')
# # 	loss = np.mean(mse)

# # 	return {'status': 'ok',
# # 			'loss': loss,
# # 			'mae': np.mean(mae),
# #             'params': params}

# # opt_f = partial(optimize, df_train=df_train, df_test=target)

# # trials = Trials()
# # best = fmin(fn=opt_f,
# #             space=space,
# #             algo=tpe.suggest,
# #             max_evals=50,
# #             trials=trials)

# # best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
# # print(best_params)
# # best_loss = trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
# # print(best_loss)
# # best_mae = trials.results[np.argmin([r['loss'] for r in trials.results])]['mae']
# # print(best_mae)
# '''
# {'activations_1': 'sigmoid', 'activations_2': 'tanh', 'batch': 782, 'dropout': 0.4500961021391858, 'nodes_1': 120, 'nodes_2': 118, 'optimizer': 'nadam'}
# 1.305828094449767
# 0.887451
# '''