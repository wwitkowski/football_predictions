from data_operations import FootballStats
from data_operations import TeamStats
from datetime import datetime, timedelta
from analysis import Distribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
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

# ts = TeamStats(decay_factor=0, num_of_matches=10)
# sim_df = pd.DataFrame()
# counter = 0

# with FootballStats() as stats:
# 	for date in stats.data.date.unique():
# 		if date < '2017-01-01':
# 			continue
# 		elif date == DATE:
# 			break
# 		try:
# 			date_data = ts.get_past_average(stats.data, date=date, exclude_features=EXCLUDE_FEATURES)
# 			sim_df = pd.concat([sim_df, date_data], sort=False, ignore_index=True)
# 		except ValueError as e:
# 			print(e)
# 			print(stats.data[(stats.data.date == date)].league)
# 		counter += 1
# 		print('Calculating stats... {0:.2f}% done'.format((counter / len(stats.data.date.unique())) * 100), end="\r")

#sim_df.to_csv('training_data.csv', index=False)

drop_features = ['score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'adj_score1',
	'adj_score2', 'HomeTeam', 'AwayTeam', 'FTR', 'shots1', 'shots2', 'shotsot1',
	'shotsot2', 'fouls1', 'fouls2', 'corners1', 'corners2', 'yellow1', 'yellow2',
	'red1', 'red2', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'avg_xg1',
	'avg_xg2', 'adj_avg_xg1', 'adj_avg_xg2', 'pts1', 'pts2', 'xpts1', 'xpts2',
	'xgshot1', 'xgshot2', 'convrate1', 'convrate2', 'cards1', 'cards2', 'prob1',
	'prob2', 'probtie', 'proj_score1', 'proj_score2', 'season', 'league_id',
	'importance1_away', 'importance2_away', 'importance1_home', 'importance2_home',
	'spi1_home', 'spi1_away']

df = pd.read_csv('training_data.csv')
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

target = df[['score1', 'score2']]

df.drop(drop_features, axis=1, inplace=True)
print(df.isna().sum().sum())
print(df.replace([np.inf, -np.inf], np.nan).isna().sum().sum())

df_strght = pd.DataFrame()
dist = Distribution()
for column in df:
	if df[column].dtype == 'float64':
		df_strght[f'strght_{column}'] = dist.straighten(df[column] + 0.001)

# df = df_strght
# features = ['strght_adj_avg_xg1_home', 'strght_xgshot1_home', 'strght_shotsot1_home']
# df = df[features]
# print(df.describe())


df.drop(['date', 'team1', 'team2', 'league'], axis=1, inplace=True)
# features = ['adj_avg_xg2_home', 'adj_score2_home', 'avg_xg2_home', 'nsxg2_home', 'score2_home', 'xg2_home', 'xgshot2_home', 'shots2_home', 'shotsot2_home']
features = ['adj_avg_xg1_home', 'nsxg2_home', 'xgshot1_home', 'xgshot2_home', 'shotsot1_home', 'shotsot2_home']
df = df[features]

print(df.var())

for column in df:
	p_corr, _ = pearsonr(df[column], target.score1)
	sp_corr, _ = spearmanr(df[column], target.score1)
	print(f'Pearson corr {column}: {p_corr}')
	print(f'Spearman corr {column}: {sp_corr}')


X_train, X_val, y_train, y_val = train_test_split(df, target.score1, test_size=0.2, random_state=0)
X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_, tick_label=df.columns)
plt.show()
xgb.plot_importance(xgb_model)
plt.show()

cb_model = CatBoostRegressor()
cb_model.fit(X_train, y_train)
plt.bar(range(len(cb_model.feature_importances_)), cb_model.feature_importances_, tick_label=df.columns)
plt.show()

corr_matrx = df[features].corr()
sns.heatmap(corr_matrx, annot=True, cmap="YlGnBu")
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