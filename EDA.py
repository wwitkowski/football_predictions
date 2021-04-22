from data_operations import FootballStats
from data_operations import TeamStats
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DATE = datetime.now().strftime('%Y-%m-%d')
# EXCLUDE_FEATURES = ['prob1','prob2',
# 					'probtie','proj_score1',
# 					'proj_score2','MaxH', 
# 					'MaxD','MaxA',
# 					'AvgH','AvgD',
# 					'AvgA', 'league_id',
# 					'season']

# stats_decay_001_num_40 = TeamStats(decay_factor=0.001, num_of_matches=40)
# stats_decay_003_num_40 = TeamStats(decay_factor=0.001, num_of_matches=60)
# stats_decay_005_num_40 = TeamStats(decay_factor=0.001, num_of_matches=80)
# stats_decay_010_num_40 = TeamStats(decay_factor=0.003, num_of_matches=80)

# sim_df_decay_001_num_40 = pd.DataFrame()
# sim_df_decay_003_num_40 = pd.DataFrame()
# sim_df_decay_005_num_40 = pd.DataFrame()
# sim_df_decay_010_num_40 = pd.DataFrame()

# counter = 0

# with FootballStats() as stats:
# 	for date in stats.data.date.unique():
# 		if date < '2017-01-01':
# 			continue
# 		elif date == DATE:
# 			break
# 		try:
# 			date_data_decay_001_num_40 = stats_decay_001_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			sim_df_decay_001_num_40 = pd.concat([sim_df_decay_001_num_40, date_data_decay_001_num_40], sort=False, ignore_index=True)

# 			date_data_decay_003_num_40 = stats_decay_003_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			sim_df_decay_003_num_40 = pd.concat([sim_df_decay_003_num_40, date_data_decay_003_num_40], sort=False, ignore_index=True)

# 			date_data_decay_005_num_40 = stats_decay_005_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			sim_df_decay_005_num_40 = pd.concat([sim_df_decay_005_num_40, date_data_decay_005_num_40], sort=False, ignore_index=True)

# 			date_data_decay_010_num_40 = stats_decay_010_num_40.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			sim_df_decay_010_num_40 = pd.concat([sim_df_decay_010_num_40, date_data_decay_010_num_40], sort=False, ignore_index=True)
# 		except ValueError as e:
# 			pass
# 		counter += 1
# 		print('Calculating stats... {0:.2f}% done'.format((counter / len(stats.data.date.unique())) * 100), end="\r")

# sim_df_decay_001_num_40.to_csv('training_data_decay_001_num_40.csv', index=False)
# sim_df_decay_003_num_40.to_csv('training_data_decay_001_num_60.csv', index=False)
# sim_df_decay_005_num_40.to_csv('training_data_decay_001_num_80.csv', index=False)
# sim_df_decay_010_num_40.to_csv('training_data_decay_003_num_80.csv', index=False)

drop_features = ['xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1',
	'adj_score2', 'HomeTeam', 'AwayTeam', 'FTR', 'shots1', 'shots2', 'shotsot1',
	'shotsot2', 'fouls1', 'fouls2', 'corners1', 'corners2', 'yellow1', 'yellow2',
	'red1', 'red2', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'avg_xg1',
	'avg_xg2', 'adj_avg_xg1', 'adj_avg_xg2', 'pts1', 'pts2', 'xpts1', 'xpts2',
	'xgshot1', 'xgshot2', 'convrate1', 'convrate2', 'cards1', 'cards2', 'prob1',
	'prob2', 'probtie', 'proj_score1', 'proj_score2', 'season',
	'importance1_away', 'importance2_away', 'importance1_home', 'importance2_home',
	'spi1_home', 'spi1_away']

### Number of matches and weighting impact analysis	 ###

# goals_stats = ['adj_avg_xg1_home', 'adj_avg_xg2_away', 'xg1_home', 'xg2_away', 'nsxg1_home', 'nsxg2_away', 'avg_xg1_home', 'avg_xg2_away', 'xg1', 'xg2']

# df_10 = pd.read_csv('training_data_decay_000_num_20.csv')
# df_20 = pd.read_csv('training_data_decay_000_num_30.csv')
# df_30 = pd.read_csv('training_data_decay_000_num_40.csv')
# df_40 = pd.read_csv('training_data_decay_001_num_40.csv')

# target = df_10[['score1', 'score2']]

# df_10 = df_10[goals_stats]
# print(df_10.describe())
# df_20 = df_20[goals_stats]
# print(df_20.describe())
# df_30 = df_30[goals_stats]
# print(df_30.describe())
# df_40 = df_40[goals_stats]
# print(df_40.describe())


# for df in [df_10, df_20, df_30, df_40]:
# 	df_corr_pears = df.corr(method='pearson')
# 	sns.heatmap(df_corr_pears, annot=True, cmap="YlGnBu")
# 	plt.title("Pearson Correlation", fontsize =20)
# 	plt.show()

# 	df_corr_kend = df.corr(method='kendall')
# 	sns.heatmap(df_corr_kend, annot=True, cmap="PiYG")
# 	plt.title("Kendall  Correlation", fontsize =20)
# 	plt.show()

# 	df_corr_spear = df.corr(method='spearman')
# 	plt.title("Spearman Correlation", fontsize =20)
# 	sns.heatmap(df_corr_spear, annot=True)
# 	plt.show()

# adj_avg = df_10[['adj_avg_xg1_home', 'adj_avg_xg2_away']].tail(df_40.shape[0])
# adj_avg['adj_avg_xg1_20'] = df_20[['adj_avg_xg1_home']].tail(df_40.shape[0])
# adj_avg['adj_avg_30'] = df_30[['adj_avg_xg1_home']].tail(df_40.shape[0])
# adj_avg['adj_avg_40'] = df_40[['adj_avg_xg1_home']]

# df_corr_pears = adj_avg.corr(method='pearson')
# sns.heatmap(df_corr_pears, annot=True, cmap="YlGnBu")
# plt.title("Pearson Correlation", fontsize =20)
# plt.show()

# df_corr_kend = adj_avg.corr(method='kendall')
# sns.heatmap(df_corr_kend, annot=True, cmap="PiYG")
# plt.title("Kendall  Correlation", fontsize =20)
# plt.show()

# df_corr_spear = adj_avg.corr(method='spearman')
# plt.title("Spearman Correlation", fontsize =20)
# sns.heatmap(df_corr_spear, annot=True)
# plt.show()

### Feature importance analysis ###

df = pd.read_csv('training_data_decay_000_num_20.csv')
df = df[(df['spi1'] > 60 and df['spi1'] < 80)]
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

target = df[['score1', 'score2']]

df.drop(drop_features, axis=1, inplace=True)
print(df.isna().sum().sum())
print(df.replace([np.inf, -np.inf], np.nan).isna().sum().sum())

# df_strght = pd.DataFrame()
# dist = Distribution()
# for column in df:
# 	if df[column].dtype == 'float64':
# 		df_strght[f'strght_{column}'] = dist.straighten(df[column] + 0.001)
# df = df_strght
# features = ['strght_adj_avg_xg1_home', 'strght_adj_avg_xg2_home', 'strght_xgshot1_home', 'strght_xgshot2_home','strght_corners1_home', 'strght_corners2_home', 'strght_fouls1_home',
# 			'strght_adj_avg_xg1_away', 'strght_adj_avg_xg2_away','strght_xgshot1_away', 'strght_xgshot2_away','strght_corners1_away', 'strght_corners2_away', 'strght_fouls1_away',  
# 			'strght_importance1', 'strght_importance2', 'strght_xg1_similar', 'strght_xg2_similar', 'strght_H', 'strght_D', 'strght_A']

df.drop(['date', 'team1', 'team2', 'league', 'league_id'], axis=1, inplace=True)


df['adj_avg_xg1_diff'] = df['adj_avg_xg1_home'] - df['adj_avg_xg2_home']
df['xgshot_diff'] = df['xgshot1_home'] - df['xgshot2_home']
df['corners_diff'] = df['corners1_home'] - df['corners2_home']

df['adj_avg_xg1_diff2'] = df['adj_avg_xg1_home'] - df['adj_avg_xg1_away']
df['xgshot_diff2'] = df['xgshot1_home'] - df['xgshot1_away']
df['corners_diff2'] = df['corners1_home'] - df['corners1_away']


df['spi_diff'] = df['spi1'] - df['spi2']
df['importance_diff'] = df['importance1'] - df['importance2']

features = ['avg_xg1_home', 'avg_xg2_home', 'xgshot1_home', 'corners1_home', 'corners2_home',
			'avg_xg1_away', 'avg_xg2_away', 'xgshot1_away', 'corners1_away', 'corners2_away',
			'spi_diff', 'importance1', 'importance2', 'xg1_similar', 'xg2_similar']
df = df[features]
# sns.pairplot(df)
# plt.show()


X_train, X_val, y_train, y_val = train_test_split(df, target.score1, test_size=0.2, random_state=0)
# X_train = np.array(X_train)
# X_val = np.array(X_val)
# y_train = np.array(y_train)
# y_val = np.array(y_val)

# Scale data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(scaler.transform(X_val.values), columns=X_val.columns, index=X_val.index)

# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

print(X_train.var())
print(X_train.describe())
# sns.pairplot(X_train)
# plt.show()

for column in X_train:
	p_corr, _ = pearsonr(X_train[column], y_train)
	sp_corr, _ = spearmanr(X_train[column], y_train)
	print(f'Pearson corr {column}: {p_corr}')
	print(f'Spearman corr {column}: {sp_corr}')

corr_matrx = X_train[features].corr()
sns.heatmap(corr_matrx, annot=True, cmap="YlGnBu")
plt.show()


xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_, tick_label=df.columns)
plt.show()
xgb.plot_importance(xgb_model)
plt.show()

cb_model = CatBoostRegressor()
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
plt.bar(range(len(cb_model.feature_importances_)), cb_model.feature_importances_, tick_label=df.columns)
plt.show()



print('xg_boost')
y_pred = xgb_model.predict(X_val)

mae = tf.keras.metrics.MeanAbsoluteError()
mae.update_state(y_pred, y_val)
print(f'MAE: {mae.result().numpy()}')

mse = tf.keras.metrics.RootMeanSquaredError()
mse.update_state(y_pred, y_val)
print(f'MSE: {mse.result().numpy()}')


print('catboost')
y_pred = cb_model.predict(X_val)

mae = tf.keras.metrics.MeanAbsoluteError()
mae.update_state(y_pred, y_val)
print(f'MAE: {mae.result().numpy()}')

mse = tf.keras.metrics.RootMeanSquaredError()
mse.update_state(y_pred, y_val)
print(f'MSE: {mse.result().numpy()}')