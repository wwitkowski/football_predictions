from data_operations import FootballStats
from data_operations import TeamStats, PoissonInflatedModel
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

# DATE = datetime.now().strftime('%Y-%m-%d')
# EXCLUDE_FEATURES = ['prob1','prob2',
# 					'probtie','proj_score1',
# 					'proj_score2','MaxH', 
# 					'MaxD','MaxA',
# 					'AvgH','AvgD',
# 					'AvgA', 'league_id',
# 					'season']

# stats_decay_001_num_40 = TeamStats(decay_factor=0.001, num_of_matches=40)
# # stats_decay_003_num_40 = TeamStats(decay_factor=0.001, num_of_matches=60)
# # stats_decay_005_num_40 = TeamStats(decay_factor=0.001, num_of_matches=80)
# # stats_decay_010_num_40 = TeamStats(decay_factor=0.003, num_of_matches=80)

# sim_df_train_decay_001_num_40 = pd.DataFrame()
# # sim_df_train_decay_003_num_40 = pd.DataFrame()
# # sim_df_train_decay_005_num_40 = pd.DataFrame()
# # sim_df_train_decay_010_num_40 = pd.DataFrame()

# counter = 0

# with FootballStats() as stats:
# 	for date in stats.data.date.unique():
# 		if date < '2017-01-01':
# 			continue
# 		elif date == DATE:
# 			break
# 		try:
# 			date_data_decay_001_num_40 = stats_decay_001_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			sim_df_train_decay_001_num_40 = pd.concat([sim_df_train_decay_001_num_40, date_data_decay_001_num_40], sort=False, ignore_index=True)

# 			# date_data_decay_003_num_40 = stats_decay_003_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			# sim_df_train_decay_003_num_40 = pd.concat([sim_df_train_decay_003_num_40, date_data_decay_003_num_40], sort=False, ignore_index=True)

# 			# date_data_decay_005_num_40 = stats_decay_005_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			# sim_df_train_decay_005_num_40 = pd.concat([sim_df_train_decay_005_num_40, date_data_decay_005_num_40], sort=False, ignore_index=True)

# 			# date_data_decay_010_num_40 = stats_decay_010_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			# sim_df_train_decay_010_num_40 = pd.concat([sim_df_train_decay_010_num_40, date_data_decay_010_num_40], sort=False, ignore_index=True)
# 		except ValueError as e:
# 			pass
# 		counter += 1
# 		print('Calculating stats... {0:.2f}% done'.format((counter / len(stats.data.date.unique())) * 100), end="\r")

# sim_df_train_decay_001_num_40.to_csv('training_data_decay_001_num_40_luck.csv', index=False)
# sim_df_train_decay_003_num_40.to_csv('training_data_decay_001_num_60.csv', index=False)
# sim_df_train_decay_005_num_40.to_csv('training_data_decay_001_num_80.csv', index=False)
# sim_df_train_decay_010_num_40.to_csv('training_data_decay_003_num_80.csv', index=False)

# drop_features = ['score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1', 'adj_score2', 'FTR',
# 	'HomeTeam', 'AwayTeam', 'shots1', 'shots2', 'shotsot1',
# 	'shotsot2', 'fouls1', 'fouls2', 'corners1', 'corners2', 'yellow1', 'yellow2',
# 	'red1', 'red2', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'avg_xg1',
# 	'avg_xg2', 'adj_avg_xg1', 'adj_avg_xg2', 'pts1', 'pts2', 'xpts1', 'xpts2',
# 	'xgshot1', 'xgshot2', 'convrate1', 'convrate2', 'cards1', 'cards2', 'prob1',
# 	'prob2', 'probtie', 'proj_score1', 'proj_score2', 'season',
# 	'importance1_away', 'importance2_away', 'importance1_home', 'importance2_home',
# 	'spi1_home', 'spi1_away']

### Number of matches and weighting impact analysis	 ###

# goals_stats = ['adj_avg_xg1_home', 'adj_avg_xg2_away', 'xg1_home', 'xg2_away', 'nsxg1_home', 'nsxg2_away', 'avg_xg1_home', 'avg_xg2_away', 'xg1', 'xg2']

# df_train_10 = pd.read_csv('training_data_decay_000_num_20.csv')
# df_train_20 = pd.read_csv('training_data_decay_000_num_30.csv')
# df_train_30 = pd.read_csv('training_data_decay_000_num_40.csv')
# df_train_40 = pd.read_csv('training_data_decay_001_num_40.csv')

# target = df_train_10[['score1', 'score2']]

# df_train_10 = df_train_10[goals_stats]
# print(df_train_10.describe())
# df_train_20 = df_train_20[goals_stats]
# print(df_train_20.describe())
# df_train_30 = df_train_30[goals_stats]
# print(df_train_30.describe())
# df_train_40 = df_train_40[goals_stats]
# print(df_train_40.describe())


# for df_train in [df_train_10, df_train_20, df_train_30, df_train_40]:
# 	df_train_corr_pears = df_train.corr(method='pearson')
# 	sns.heatmap(df_train_corr_pears, annot=True, cmap="YlGnBu")
# 	plt.title("Pearson Correlation", fontsize =20)
# 	plt.show()

# 	df_train_corr_kend = df_train.corr(method='kendall')
# 	sns.heatmap(df_train_corr_kend, annot=True, cmap="PiYG")
# 	plt.title("Kendall  Correlation", fontsize =20)
# 	plt.show()

# 	df_train_corr_spear = df_train.corr(method='spearman')
# 	plt.title("Spearman Correlation", fontsize =20)
# 	sns.heatmap(df_train_corr_spear, annot=True)
# 	plt.show()

# adj_avg = df_train_10[['adj_avg_xg1_home', 'adj_avg_xg2_away']].tail(df_train_40.shape[0])
# adj_avg['adj_avg_xg1_20'] = df_train_20[['adj_avg_xg1_home']].tail(df_train_40.shape[0])
# adj_avg['adj_avg_30'] = df_train_30[['adj_avg_xg1_home']].tail(df_train_40.shape[0])
# adj_avg['adj_avg_40'] = df_train_40[['adj_avg_xg1_home']]

# df_train_corr_pears = adj_avg.corr(method='pearson')
# sns.heatmap(df_train_corr_pears, annot=True, cmap="YlGnBu")
# plt.title("Pearson Correlation", fontsize =20)
# plt.show()

# df_train_corr_kend = adj_avg.corr(method='kendall')
# sns.heatmap(df_train_corr_kend, annot=True, cmap="PiYG")
# plt.title("Kendall  Correlation", fontsize =20)
# plt.show()

# df_train_corr_spear = adj_avg.corr(method='spearman')
# plt.title("Spearman Correlation", fontsize =20)
# sns.heatmap(df_train_corr_spear, annot=True)
# plt.show()

### Feature importance analysis ###

df = pd.read_csv('training_data\\training_data_decay_001_num_40_luck.csv')

# df_train = df_train[df_train['league'] == 'Barclays Premier League']
# df_train = df_train[df_train['spi1'] > 80]
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

test_matches_number = 1000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()

print(df_test.head(1))
print(df_train.tail(1))

target = df_train[['score1', 'score2']]

df_train.drop(['score1', 'score2'], axis=1, inplace=True)
print(df_train.isna().sum().sum())
print(df_train.replace([np.inf, -np.inf], np.nan).isna().sum().sum())

# df_train_strght = pd.DataFrame()
# dist = Distribution()
# for column in df_train:
# 	if df_train[column].dtype == 'float64':
# 		df_train_strght[f'strght_{column}'] = dist.straighten(df_train[column] + 0.001)
# df_train = df_train_strght
# features = ['strght_adj_avg_xg1_home', 'strght_adj_avg_xg2_home', 'strght_xgshot1_home', 'strght_xgshot2_home','strght_corners1_home', 'strght_corners2_home', 'strght_fouls1_home',
# 			'strght_adj_avg_xg1_away', 'strght_adj_avg_xg2_away','strght_xgshot1_away', 'strght_xgshot2_away','strght_corners1_away', 'strght_corners2_away', 'strght_fouls1_away',  
# 			'strght_importance1', 'strght_importance2', 'strght_xg1_similar', 'strght_xg2_similar', 'strght_H', 'strght_D', 'strght_A']

df_train.drop(['date', 'team1', 'team2', 'league', 'league_id'], axis=1, inplace=True)

df_train['adj_avg_xg_diff_home'] = df_train['adj_avg_xg1_home'] - df_train['adj_avg_xg2_home']
df_train['adj_avg_xg_diff_away'] = df_train['adj_avg_xg1_away'] - df_train['adj_avg_xg2_away']
df_train['xgshot_diff'] = df_train['xgshot1_home'] - df_train['xgshot2_home']
df_train['corners_diff'] = df_train['corners1_home'] - df_train['corners2_home']

df_train['adj_avg_xg1_diff2'] = df_train['adj_avg_xg1_home'] - df_train['adj_avg_xg1_away']
df_train['xgshot_diff2'] = df_train['xgshot1_home'] - df_train['xgshot1_away']
df_train['corners_diff2'] = df_train['corners1_home'] - df_train['corners1_away']

df_train['spi_diff'] = df_train['spi1'] - df_train['spi2']
df_train['importance_diff'] = df_train['importance1'] - df_train['importance2']

# print(df_train[['importance_diff']].describe())


# goals_pos_luck = df_train[['xg1']].loc[df_train['importance_diff'] > 30]
# goals_neg_luck = df_train[['xg1']].loc[df_train['importance_diff'] <= 30]
# means = (goals_pos_luck.mean().values, goals_neg_luck.mean().values)
# print(goals_pos_luck.describe())
# print(goals_neg_luck.describe())

# df_train2 = df_train[['xg1', 'xg2', 'shotsot1_home', 'shotsot2_away', 'FTR']]
# sns.pairplot(df_train2, hue='FTR')
# plt.show()
# 




features = ['avg_xg1_home', 'avg_xg2_home', 'xgshot1_home', 'xgshot2_home', 'shots1_home', 'shotsot1_home',  'shots2_home', 'shotsot2_home', 
			'corners1_home', 'corners2_home', 'fouls1_home', 'fouls2_home', 'cards1_home', 'cards2_home',
			'xpts1_home', 'convrate1_home', 'convrate2_home',
			'avg_xg1_away', 'avg_xg2_away', 'xgshot1_away', 'xgshot2_away', 'shots1_away', 'shotsot1_away',  'shots2_away', 'shotsot2_away',
			'corners1_away', 'corners2_away', 'fouls1_away', 'fouls2_away', 'cards1_away', 'cards2_away', 
			'xpts1_away', 'convrate1_away', 'convrate2_away',
			'spi1', 'spi2', 
			'importance1', 'importance2', 'xg1_similar', 'xg2_similar', 'past_avg_luck', 'past_avg_luck_away', 'A', 'D', 'H',
			'spi_diff', 'importance_diff']

score1_features = ['avg_xg1_home', 'xgshot1_home',
			'shots1_home', 'shotsot1_home', 'shotsot2_home', 'shots2_home', 'corners1_home',
			'corners2_home', 'fouls1_home',
			'xpts1_home', 'convrate1_home',
			'avg_xg1_away',
			'shots1_away', 'shotsot1_away', 'shots2_away', 'shotsot2_away', 'corners1_away',
			'corners2_away',
			'xpts1_away',
			'convrate1_away',
			'importance1', 'importance2',
			'xg1_similar', 'xg2_similar',
			'D', 'spi_diff']

df_train = df_train[score1_features]



X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.2, random_state=0)


# Scale data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(scaler.transform(X_val.values), columns=X_val.columns, index=X_val.index)





# for column in X_train:
# 	p_corr, _ = pearsonr(X_train[column], y_train)
# 	sp_corr, _ = spearmanr(X_train[column], y_train)
# 	print(f'Pearson corr {column}: {p_corr}')
# 	print(f'Spearman corr {column}: {sp_corr}')

# corr_matrx = X_train[features].corr()
# sns.heatmap(corr_matrx, annot=True, cmap="YlGnBu")
# plt.show()

# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train)
# X_val_pca = pca.transform(X_val)
# plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, alpha=0.5,
#         align='center', label='individual explained variance')

# plt.show()
# print(sum(pca.explained_variance_ratio_))



# alpha = np.linspace(0.001, 0.01, 101)
# print(alpha)
# search = GridSearchCV(estimator=Lasso(), param_grid={'alpha': alpha}, cv=5, scoring='neg_mean_absolute_error', verbose=3)
# search.fit(X_train, y_train)

# lasso = Lasso(alpha=0.001)
# lasso.fit(X_train, y_train)

# lasso_y_pred = lasso.predict(X_val)
# print(f"MAE: {mean_absolute_error(y_val, lasso_y_pred, multioutput='raw_values')}")
# print(f"MSE: {mean_squared_error(y_val, lasso_y_pred, multioutput='raw_values')}")


# print('xg_boost')
# xgb_model_score1 = xgb.XGBRegressor()
# xgb_model_score1.fit(X_train, y_train.score1)
# xgb_model_score2 = xgb.XGBRegressor()
# xgb_model_score2.fit(X_train, y_train.score2)

# xgb_score1_pred = xgb_model_score1.predict(X_val)
# xgb_score2_pred = xgb_model_score2.predict(X_val)

# print(f"MAE score1: {mean_absolute_error(y_val.score1, xgb_score1_pred)}")
# print(f"MAE score2: {mean_absolute_error(y_val.score2, xgb_score2_pred)}")

# print('cat_boost')
# cb_model_score1 = CatBoostRegressor()
# cb_model_score1.fit(X_train, y_train.score1, eval_set=(X_val, y_val.score1), early_stopping_rounds=10)
# cb_model_score2 = CatBoostRegressor()
# cb_model_score2.fit(X_train, y_train.score2, eval_set=(X_val, y_val.score2), early_stopping_rounds=10)

# cb_score1_pred = cb_model_score1.predict(X_val)
# cb_score2_pred = cb_model_score2.predict(X_val)

# print(f"MAE score1: {mean_absolute_error(y_val.score1, cb_score1_pred)}")
# print(f"MAE score2: {mean_absolute_error(y_val.score2, cb_score2_pred)}")

print('neural network')
# activations = ('relu', 'relu')
# nodes = (4, 8)
# nn_model = NeuralNetworkModel()
# nn_model.build(n_features=X_train.shape[1],
# 				activations=activations,
# 				nodes=nodes)

# history = nn_model.train(X_train.values, y_train.values, X_val.values, y_val.values, 
# 						verbose=1, batch_size=512, epochs=1000)


best_model = NeuralNetworkModel('nn_model')

df_test['spi_diff'] = df_test['spi1'] - df_test['spi2']
X_test = df_test[score1_features]
X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns, index=X_test.index)
y_test = df_test[['score1', 'score2']]
nn_y_pred = best_model.predict(X_test.values)

pred_home_goals = nn_y_pred[:, 0]
pred_away_goals = nn_y_pred[:, 1]
foot_poisson = FootballPoissonModel(data=df)
home_win, draw, away_win = foot_poisson.predict_chances(pred_home_goals, pred_away_goals)

df_predictions = df_test[['date', 'league', 'team1', 'team2', 'score1', 'score2', 'FTR', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 
						  'FTR', 'shots1', 'shots2', 'shotsot1', 'shotsot2', 'fouls1', 'fouls2', 'corners1', 'corners2', 
						  'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'xg1_similar', 'xg2_similar', 'H', 'D', 'A']]

predictions = pd.DataFrame(data={'score1_pred': pred_home_goals, 'score2_pred': pred_away_goals,
								 'homewin_pred': home_win, 'draw_pred': draw, 'awaywin_pred': away_win})

print(predictions)

# p = PoissonInflatedModel(p=0.08, teta=0.14)
# predictions_infalted = pd.DataFrame(data={'score1_pred': pred_home_goals, 'score2_pred': pred_away_goals})
# predictions_infalted['homewin_pred'], predictions_infalted['draw_pred'], predictions_infalted['awaywin_pred'] = zip(*predictions_infalted.apply(lambda row: p.win_probabilities(row.score1_pred, row.score2_pred), axis=1))
# print(predictions_infalted.describe())

df_predictions = pd.concat([df_predictions.reset_index(drop=True), predictions], axis=1)
print()
df_predictions.to_csv('predictions.csv', index=False)






# print(f"MAE: {mean_absolute_error(y_val, nn_y_pred, multioutput='raw_values')}")
# print(f"MSE: {mean_squared_error(y_val, nn_y_pred, multioutput='raw_values')}")


# print('Neural network KFOOLD CV')

# batches = [64, 128, 256, 512]

# kf = KFold(n_splits=8)
# for batches_setting in batches:
# 	print(f'Batches setting: {batches_setting}')
# 	df_train_copy = df_train.copy()
# 	target_copy = target.copy()
# 	scores = []
# 	for train, test in kf.split(df_train_copy):
# 		scaler = StandardScaler()
# 		X_train = scaler.fit_transform(df_train_copy.iloc[train])
# 		X_val = scaler.transform(df_train_copy.iloc[test])

# 		y_train = target_copy.iloc[train].values
# 		y_val = target_copy.iloc[test].values

# 		nn_model = NeuralNetworkModel()
# 		nn_model.build(n_features=X_train.shape[1],
# 							activations=activations,
# 							nodes=nodes)

# 		history = nn_model.train(X_train, y_train, X_val, y_val, 
# 						   			 verbose=0, batch_size=batches_setting, epochs=300)
# 		scores.append(np.min(history.history['val_mae']))
# 	print(f'Average min loss: {np.mean(scores)}, Std dev: {np.std(scores)}')

