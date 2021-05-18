from data_operations import FootballStats
from data_operations import TeamStats
from data_operations import Bookmaker
from datetime import datetime, timedelta
from analysis import Distribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind

import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

from models import NeuralNetworkModel, FootballPoissonModel

from functools import partial
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

# DATE = datetime.now().strftime('%Y-%m-%d')
# EXCLUDE_FEATURES = ['prob1', 'prob2', 'probtie', 
# 					'proj_score1', 'proj_score2', 'MaxH', 'MaxD', 'MaxA',
# 					'AvgH', 'AvgD', 'AvgA', 'league_id', 'season']

# ts = TeamStats(decay_factor=0.00325, num_of_matches=40, min_matches=8)
# final_data = pd.DataFrame()
# counter = 0

# with FootballStats() as stats:
# 	for date in stats.data.date.unique():
# 		if date < '2017-01-01':
# 			continue
# 		elif date == DATE:
# 			break
# 		try:
# 			date_data = ts.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			final_data = pd.concat([final_data, date_data], sort=False, ignore_index=True)
# 		except ValueError as e:
# 			pass
# 		counter += 1
# 		print('Calculating stats... {0:.2f}% done'.format((counter / len(stats.data.date.unique())) * 100), end="\r")

# final_data.to_csv('training_data\\training_data_decay00325_num40_wluck.csv', index=False)


df = pd.read_csv('training_data\\training_data_decay00325_num40_wluck.csv')
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

df['spi_diff'] = df['spi1'] - df['spi2']
df['convrate_diff'] = df['convrate1_home'] + df['convrate2_away']
df['avg_xg_diff'] = df['avg_xg1_home'] + df['avg_xg2_away']
df['avg_xg_diff_home'] = df['avg_xg1_home'] - df['avg_xg2_home']
df['avg_xg_diff_away'] = df['avg_xg1_away'] - df['avg_xg2_away']

test_matches_number = 2000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()

target = df_train[['score1', 'score2']]
df_train.drop(['score1', 'score2'], axis=1, inplace=True)
print(f'Number of NA values: {df_train.isna().sum().sum()}')
print(f'Number of INF values: {df_train.replace([np.inf, -np.inf], np.nan).isna().sum().sum()}')




# features = ['spi1','importance1','score1_home','score2_home','xg1_home','xg2_home','nsxg1_home','nsxg2_home',
# 			'shots1_home','shots2_home','shotsot1_home','shotsot2_home','corners1_home','corners2_home',
# 			'avg_xg1_home','avg_xg2_home','adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home','xdraw_home',
# 			'xwin2_home','xpts1_home','xpts2_home','xgshot1_home','convrate1_home','convrate2_home','spi2',
# 			'importance2','score1_away','score2_away','xg1_away','xg2_away','nsxg1_away','nsxg2_away',
# 			'shots1_away','shots2_away','shotsot1_away','shotsot2_away','corners1_away','corners2_away',
# 			'avg_xg1_away','avg_xg2_away','adj_avg_xg1_away','adj_avg_xg2_away','xwin1_away','xdraw_away',
# 			'xwin2_away','xpts1_away','xpts2_away','xgshot1_away','convrate1_away','convrate2_away','A','H',
# 			'score1_similar','score2_similar','xg1_similar','xg2_similar']

features = ['importance1',
			'shots1_home','shots2_home','shotsot1_home','shotsot2_home','corners1_home','corners2_home',
			'adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home','xdraw_home',
			'xwin2_home','xpts1_home','xgshot1_home','convrate1_home','convrate2_home',
			'importance2',
			'shots1_away','shots2_away','shotsot1_away','shotsot2_away','corners1_away','corners2_away',
			'adj_avg_xg1_away','adj_avg_xg2_away','xwin1_away','xdraw_away',
			'xwin2_away','xpts1_away','xgshot1_away','convrate1_away','convrate2_away',
			'A','H','score1_similar','score2_similar',
			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away']

df_train = df_train[features]
# sns.pairplot(df_train)
# plt.show()

# df_train_strght = pd.DataFrame()
# dist = Distribution()
# for column in df_train:
# 	if df_train[column].dtype == 'float64':
# 		if df_train[column].lt(0).any():
# 			df_train_strght[f'{column}'] = df_train[column].values
# 		else:
# 			df_train_strght[f'strght_{column}'] = dist.straighten(df_train[column] + 0.001)
			# sns.distplot(df_train[column])
			# sns.distplot(df_train_strght[f'strght_{column}'], color='g')
			# plt.legend(labels=['Original distribution', 'Straightened distribution'])
			# plt.show()


# df_variance = df_train.var()
# plt.bar(np.arange(df_variance.shape[0]), df_variance, log=True)
# plt.xticks(np.arange(df_variance.shape[0]), df_train.columns.values, rotation=45, ha='right')
# plt.ylabel('Variance')
# plt.title('Feature variance')
# plt.show()


# kt_corr = [kendalltau(df_train[column], target.score1) for column in df_train]
# sp_corr = [spearmanr(df_train[column], target.score1) for column in df_train]

# rects1 = plt.bar(np.arange(df_train.shape[1])-0.2, [corr[0] for corr in kt_corr], width=0.4, color='c')
# rects2 = plt.bar(np.arange(df_train.shape[1])+0.2, [corr[0] for corr in sp_corr], width=0.4, color='m')
# padding = 0.01
# for rect in rects1:
# 	height = rect.get_height()
# 	if height > 0:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
# 				 ha='center', va='bottom', rotation=90, color='c', weight='bold')
# 	else:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2),
# 				 ha='center', va='top', rotation=90, color='c', weight='bold')

# for rect in rects2:
# 	height = rect.get_height()
# 	if height > 0:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
# 				 ha='center', va='bottom', rotation=90, color='m', weight='bold')
# 	else:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2), 
# 				 ha='center', va='top', rotation=90, color='m', weight='bold')

# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.legend(labels=['Kendall', 'Spearman'])
# plt.ylabel('Correlation')
# plt.title('Feature correlation')
# plt.show()

# kt_corr = [kendalltau(df_train[column], target.score2) for column in df_train]
# sp_corr = [spearmanr(df_train[column], target.score2) for column in df_train]

# rects1 = plt.bar(np.arange(df_train.shape[1])-0.2, [corr[0] for corr in kt_corr], width=0.4, color='c')
# rects2 = plt.bar(np.arange(df_train.shape[1])+0.2, [corr[0] for corr in sp_corr], width=0.4, color='m')
# padding = 0.01
# for rect in rects1:
# 	height = rect.get_height()
# 	if height > 0:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
# 				 ha='center', va='bottom', rotation=90, color='c', weight='bold')
# 	else:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2),
# 				 ha='center', va='top', rotation=90, color='c', weight='bold')

# for rect in rects2:
# 	height = rect.get_height()
# 	if height > 0:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
# 				 ha='center', va='bottom', rotation=90, color='m', weight='bold')
# 	else:
# 		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2), 
# 				 ha='center', va='top', rotation=90, color='m', weight='bold')

# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.legend(labels=['Kendall', 'Spearman'])
# plt.ylabel('Correlation')
# plt.title('Feature correlation')
# plt.show()

# print(df_train.xgshot1_home.describe())
# group1 = target.score1[df_train.xgshot1_home >= 0.11]
# group2 = target.score1[df_train.xgshot1_home < 0.11]
# print(group1.mean(), group2.mean())
# stat, p_val = ttest_ind(group1.values, group2.values)#, equal_var=False)
# print(f'T-Test statistic: {stat}, p value: {p_val}')
# if p_val < 0.05:
# 	print('Means are different')

# print(df_train.xgshot2_away.describe())
# group1 = target.score1[df_train.xgshot2_away >= 0.11]
# group2 = target.score1[df_train.xgshot2_away < 0.11]
# print(group1.mean(), group2.mean())
# stat, p_val = ttest_ind(group1.values, group2.values)#, equal_var=False)
# print(f'T-Test statistic: {stat}, p value: {p_val}')
# if p_val < 0.05:
# 	print('Means are different')



X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.3, random_state=0)
# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_val = scaler.transform(X_val.values)

# regr_score1 = RandomForestRegressor(n_estimators=100)
# regr_score1.fit(X_train, y_train.score1)

# xgb_model_score1 = xgb.XGBRegressor()
# xgb_model_score1.fit(X_train, y_train.score1)

# cb_model_score1 = CatBoostRegressor()
# cb_model_score1.fit(X_train, y_train.score1, eval_set=(X_val, y_val.score1), early_stopping_rounds=10)


# padding=0.001
# rects = plt.bar(np.arange(len(regr_score1.feature_importances_)), regr_score1.feature_importances_, width=0.8, color='c')
# for rect in rects:
# 	height = rect.get_height()
# 	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
# 			 ha='center', va='bottom', rotation=90, color='c', weight='bold')
# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.show()

# rects = plt.bar(np.arange(len(xgb_model_score1.feature_importances_)), xgb_model_score1.feature_importances_, width=0.8, color='m')
# for rect in rects:
# 	height = rect.get_height()
# 	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
# 			 ha='center', va='bottom', rotation=90, color='m', weight='bold')
# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.show()

# padding=0.0001
# rects = plt.bar(np.arange(len(cb_model_score1.get_feature_importance(Pool(X_train, y_train.score1), type='LossFunctionChange'))), 
# 				cb_model_score1.get_feature_importance(Pool(X_train, y_train.score1), type='LossFunctionChange'), 
# 				width=0.8, color='y')
# for rect in rects:
# 	height = rect.get_height()
# 	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 4), 
# 			 ha='center', va='bottom', rotation=90, color='y', weight='bold')
# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.show()



# regr_score2 = RandomForestRegressor(n_estimators=100)
# regr_score2.fit(X_train, y_train.score2)

# xgb_model_score2 = xgb.XGBRegressor()
# xgb_model_score2.fit(X_train, y_train.score2)

# cb_model_score2 = CatBoostRegressor()
# cb_model_score2.fit(X_train, y_train.score2, eval_set=(X_val, y_val.score2), early_stopping_rounds=10)


# padding=0.001
# rects = plt.bar(np.arange(len(regr_score2.feature_importances_)), regr_score2.feature_importances_, width=0.8, color='c')
# for rect in rects:
# 	height = rect.get_height()
# 	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
# 			 ha='center', va='bottom', rotation=90, color='c', weight='bold')
# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.show()

# rects = plt.bar(np.arange(len(xgb_model_score2.feature_importances_)), xgb_model_score2.feature_importances_, width=0.8, color='m')
# for rect in rects:
# 	height = rect.get_height()
# 	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
# 			 ha='center', va='bottom', rotation=90, color='m', weight='bold')
# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.show()

# padding=0.0001
# rects = plt.bar(np.arange(len(cb_model_score2.get_feature_importance(Pool(X_train, y_train.score2), type='LossFunctionChange'))), 
# 				cb_model_score2.get_feature_importance(Pool(X_train, y_train.score2), type='LossFunctionChange'), 
# 				width=0.8, color='y')
# for rect in rects:
# 	height = rect.get_height()
# 	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 4), 
# 			 ha='center', va='bottom', rotation=90, color='y', weight='bold')
# plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
# plt.show()




print('neural network')
activations = ('sigmoid', 'sigmoid')
nodes = (394, 118)
nn_model = NeuralNetworkModel()
nn_model.build(n_features=X_train.shape[1],
				optimizer='rmsprop',
				dropout=0.42,
				activations=activations,
				nodes=nodes)

print(nn_model.summary)

history = nn_model.train(X_train, y_train.values, X_val, y_val.values, 
						verbose=0, batch_size=782, epochs=500)


best_model = NeuralNetworkModel('nn_model')

X_test = df_test[features]
X_test = scaler.transform(X_test.values)
y_test = df_test[['score1', 'score2']]
nn_y_pred = best_model.predict(X_test)

pred_home_goals = nn_y_pred[:, 0]
pred_away_goals = nn_y_pred[:, 1]
foot_poisson = FootballPoissonModel()
home_win, draw, away_win = foot_poisson.predict_chances(pred_home_goals, pred_away_goals)

df_predictions = df_test[['date', 'league', 'team1', 'team2', 'score1', 'score2', 'FTR', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 
						  'shots1', 'shots2', 'shotsot1', 'shotsot2', 'fouls1', 'fouls2', 'corners1', 'corners2', 
						  'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'xg1_similar', 'xg2_similar', 'H', 'D', 'A']]

predictions = pd.DataFrame(data={'score1_pred': pred_home_goals, 'score2_pred': pred_away_goals,
								 'homewin_pred': home_win, 'draw_pred': draw, 'awaywin_pred': away_win})


df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
print(df_predictions.FTR)

book = Bookmaker(df_predictions, stake=5)
book.calculate()
#print(df_predictions[['FTR', 'BET', 'prediction_odds', 'bet_return']].head(20))
print(df_predictions[['bet_return']].sum().values)


# print('Neural network KFOOLD CV')

# space = {
# 		'batch': scope.int(hp.quniform('batch', 32, 2048, 2)),
# 		'activations_1': hp.choice('activations_1', ['relu', 'tanh', 'sigmoid']),
# 		'activations_2': hp.choice('activations_2', ['relu', 'tanh', 'sigmoid']),
# 		#'activations_3': hp.choice('activations_3', ['relu', 'tanh', 'sigmoid']),
# 		'nodes_1': scope.int(hp.qloguniform('nodes_1', np.log(2), np.log(512), 2)),
# 		'nodes_2': scope.int(hp.qloguniform('nodes_2', np.log(2), np.log(512), 2)),
# 		#'nodes_3': scope.int(hp.qloguniform('nodes_3', np.log(2), np.log(512), 2)),
# 		'dropout': hp.uniform('dropout', 0.01, 0.5),
# 		'optimizer': hp.choice('optimizer', ['adam', 'nadam', 'sgd', 'rmsprop'])
# 		}


# def optimize(params, df_train, df_test):

# 	kf = KFold(n_splits=8)
# 	df_train_copy = df_train.copy()
# 	target_copy = df_test.copy()
# 	mae = []
# 	mse = []
# 	for train, test in kf.split(df_train_copy):
# 		scaler = StandardScaler()
# 		X_train = scaler.fit_transform(df_train_copy.iloc[train])
# 		X_val = scaler.transform(df_train_copy.iloc[test])

# 		y_train = target_copy.iloc[train].values
# 		y_val = target_copy.iloc[test].values

# 		nn_model = NeuralNetworkModel()
# 		nn_model.build(n_features=X_train.shape[1],
# 						optimizer=params['optimizer'],
# 						dropout=params['dropout'],
# 						activations=(params['activations_1'], params['activations_2']),
# 						nodes=(params['nodes_1'], params['nodes_2']))

# 		history = nn_model.train(X_train, y_train, X_val, y_val,
# 							   			 verbose=0,
# 							   			 batch_size=params['batch'], 
# 							   			 epochs=500)

# 		mse.append(np.min(history.history['val_loss']))
# 		mae.append(np.min(history.history['val_mae']))
# 	#print(f'Average min loss: {np.mean(mae)}, Std dev: {np.std(mae)}')
# 	loss = np.mean(mse)

# 	return {'status': 'ok',
# 			'loss': loss,
# 			'mae': np.mean(mae),
#             'params': params}

# opt_f = partial(optimize, df_train=df_train, df_test=target)

# trials = Trials()
# best = fmin(fn=opt_f,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=50,
#             trials=trials)

# best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
# print(best_params)
# best_loss = trials.results[np.argmin([r['loss'] for r in trials.results])]['loss']
# print(best_loss)
# best_mae = trials.results[np.argmin([r['loss'] for r in trials.results])]['mae']
# print(best_mae)
'''
{'activations_1': 'sigmoid', 'activations_2': 'sigmoid', 'batch': 1454, 'dropout': 0.42000439850998783, 'nodes_1': 394, 'nodes_2': 118, 'optimizer': 'rmsprop'}
1.2832087278366089
0.8729606
'''