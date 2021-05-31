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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from models import NeuralNetworkModel, FootballPoissonModel

### Data creation
# DATE = datetime.now().strftime('%Y-%m-%d')
# EXCLUDE_FEATURES = ['prob1', 'prob2', 'probtie', 
# 					'proj_score1', 'proj_score2',
# 					'importance1', 'importance2',
# 					'MaxH', 'MaxD', 'MaxA',
# 					'AvgH', 'AvgD', 'AvgA', 
# 					'league_id', 'season',
# 					'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5']

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

### Data load
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
df = pd.read_csv('training_data\\training_data_decay00325_num40_wluck.csv')
df.importance1.fillna(value=df.importance1.mean(), inplace=True)
df.importance2.fillna(value=df.importance2.mean(), inplace=True)
df.dropna(inplace=True)

df['spi_diff'] = df['spi1'] - df['spi2']
df['convrate_diff'] = df['convrate1_home'] + df['convrate2_away']
df['avg_xg_diff'] = df['avg_xg1_home'] + df['avg_xg2_away']
df['avg_xg_diff_home'] = df['avg_xg1_home'] - df['avg_xg2_home']
df['avg_xg_diff_away'] = df['avg_xg1_away'] - df['avg_xg2_away']

test_matches_number = 5000
df_test = df.tail(test_matches_number).copy()
df_train = df.head(df.shape[0] - test_matches_number).copy()


### Selected features
# features = ['spi1','importance1','score1_home','score2_home','xg1_home','xg2_home','nsxg1_home','nsxg2_home',
# 			'shots1_home','shots2_home','shotsot1_home','shotsot2_home','corners1_home','corners2_home',
# 			'avg_xg1_home','avg_xg2_home','adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home','xdraw_home',
# 			'xwin2_home','xpts1_home','xpts2_home','xgshot1_home','convrate1_home','convrate2_home','spi2',
# 			'importance2','score1_away','score2_away','xg1_away','xg2_away','nsxg1_away','nsxg2_away',
# 			'shots1_away','shots2_away','shotsot1_away','shotsot2_away','corners1_away','corners2_away',
# 			'avg_xg1_away','avg_xg2_away','adj_avg_xg1_away','adj_avg_xg2_away','xwin1_away','xdraw_away',
# 			'xwin2_away','xpts1_away','xpts2_away','xgshot1_away','convrate1_away','convrate2_away','A','H',
# 			'score1_similar','score2_similar','xg1_similar','xg2_similar']

features = ['adj_avg_xg1_home','adj_avg_xg2_home','xwin1_home', 'xdraw_home',
			'xwin2_home', 'shots1_home', 'shots2_home','shotsot1_home', 'shotsot2_home',
			'corners1_home', 'corners2_home', 'xpts1_home', 'xpts2_home', 'xgshot1_home',
			'convrate1_home',
			'adj_avg_xg1_away','adj_avg_xg2_away', 'xwin1_away', 'xdraw_away',
			'xwin2_away', 'shots1_away', 'shots2_away','shotsot1_away', 'shotsot2_away',
			'corners1_away', 'corners2_away', 'xpts1_away', 'xpts2_away',
			'A','H','score1_similar','score2_similar',
			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away',
			'score1', 'score2']

# features = ['adj_avg_xg1_home','xwin1_home',
# 			'xwin2_home', 'shots1_home','shotsot1_home',
# 			'xpts1_home', 'xpts2_home', 
# 			'adj_avg_xg1_away', 'xwin1_away',
# 			'xwin2_away', 'shots1_away', 'shotsot1_away', 
# 			'xpts1_away', 'xpts2_away',
# 			'A','H','score1_similar','score2_similar',
# 			'spi_diff', 'avg_xg_diff', 'avg_xg_diff_home', 'avg_xg_diff_away']

df_train = df_train[features]


### Boxplot - outliers
for column in df_train:
	if df_train[column].dtype == 'float64':
			print(df_train[column].mean() - 3*df_train[column].std(), df_train[column].mean() + 3*df_train[column].std())
			sns.boxplot(df_train[column])
			plt.show()

outliers = (np.abs(df_train-df_train.mean()) <= (3*df_train.std())).all(axis=1)
df_train = df_train[outliers]
print(df_train.columns.values)
target = target[outliers]


target = df_train[['score1', 'score2']]
df_train.drop(['score1', 'score2'], axis=1, inplace=True)
print(f'Number of NA values: {df_train.isna().sum().sum()}')
print(f'Number of INF values: {df_train.replace([np.inf, -np.inf], np.nan).isna().sum().sum()}')


### Correlation
df_corr = df_train.corr()
sns.heatmap(df_corr, annot=True, cmap="YlGnBu")
plt.show()


### Pairplot
# sns.pairplot(df_train)
# plt.show()


### Feature Transformation
# dist = Distribution()
# for column in df_train:
# 	if df_train[column].dtype == 'float64':
# 		if not df_train[column].lt(0).any():
# 			df_train[f'strght_{column}'] = dist.straighten(df_train[column] + 0.001)
# 			# sns.distplot(df_train[column])
# 			# sns.distplot(df_train_strght[f'strght_{column}'], color='g')
# 			# plt.legend(labels=['Original distribution', 'Straightened distribution'])
# 			# plt.show()
# df_train = df_train.loc[:, df_train.columns.str.contains('strght')]
 

### Feature variance
df_variance = df_train.var()
plt.bar(np.arange(df_variance.shape[0]), df_variance, log=True)
plt.xticks(np.arange(df_variance.shape[0]), df_train.columns.values, rotation=45, ha='right')
plt.ylabel('Variance')
plt.title('Feature variance')
plt.show()

### Score1 Spearman and Kendall correlation
kt_corr = [kendalltau(df_train[column], target.score1) for column in df_train]
sp_corr = [spearmanr(df_train[column], target.score1) for column in df_train]

print([corr[0] for corr in kt_corr])

rects1 = plt.bar(np.arange(df_train.shape[1])-0.2, [corr[0] for corr in kt_corr], width=0.4, color='c')
rects2 = plt.bar(np.arange(df_train.shape[1])+0.2, [corr[0] for corr in sp_corr], width=0.4, color='m')
padding = 0.01
for rect in rects1:
	height = rect.get_height()
	if height > 0:
		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
				 ha='center', va='bottom', rotation=90, color='c', weight='bold')
	else:
		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2),
				 ha='center', va='top', rotation=90, color='c', weight='bold')

for rect in rects2:
	height = rect.get_height()
	if height > 0:
		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
				 ha='center', va='bottom', rotation=90, color='m', weight='bold')
	else:
		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2), 
				 ha='center', va='top', rotation=90, color='m', weight='bold')

plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.legend(labels=['Kendall', 'Spearman'])
plt.ylabel('Correlation')
plt.title('Feature correlation')
plt.show()


### Score2 Spearman and Kendall correlation
kt_corr = [kendalltau(df_train[column], target.score2) for column in df_train]
sp_corr = [spearmanr(df_train[column], target.score2) for column in df_train]

rects1 = plt.bar(np.arange(df_train.shape[1])-0.2, [corr[0] for corr in kt_corr], width=0.4, color='c')
rects2 = plt.bar(np.arange(df_train.shape[1])+0.2, [corr[0] for corr in sp_corr], width=0.4, color='m')
padding = 0.01
for rect in rects1:
	height = rect.get_height()
	if height > 0:
		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
				 ha='center', va='bottom', rotation=90, color='c', weight='bold')
	else:
		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2),
				 ha='center', va='top', rotation=90, color='c', weight='bold')

for rect in rects2:
	height = rect.get_height()
	if height > 0:
		plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 2), 
				 ha='center', va='bottom', rotation=90, color='m', weight='bold')
	else:
		plt.text(rect.get_x() + rect.get_width() / 2, height - padding, round(height, 2), 
				 ha='center', va='top', rotation=90, color='m', weight='bold')

plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.legend(labels=['Kendall', 'Spearman'])
plt.ylabel('Correlation')
plt.title('Feature correlation')
plt.show()



X_train, X_val, y_train, y_val = train_test_split(df_train, target, test_size=0.3, random_state=0)
# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_val = scaler.transform(X_val.values)


### Score1 Feature Importance
regr_score1 = RandomForestRegressor(n_estimators=100)
regr_score1.fit(X_train, y_train.score1)

xgb_model_score1 = xgb.XGBRegressor()
xgb_model_score1.fit(X_train, y_train.score1)

cb_model_score1 = CatBoostRegressor()
cb_model_score1.fit(X_train, y_train.score1, eval_set=(X_val, y_val.score1), early_stopping_rounds=10)


padding=0.001
rects = plt.bar(np.arange(len(regr_score1.feature_importances_)), regr_score1.feature_importances_, width=0.8, color='c')
for rect in rects:
	height = rect.get_height()
	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
			 ha='center', va='bottom', rotation=90, color='c', weight='bold')
plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.show()

rects = plt.bar(np.arange(len(xgb_model_score1.feature_importances_)), xgb_model_score1.feature_importances_, width=0.8, color='m')
for rect in rects:
	height = rect.get_height()
	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
			 ha='center', va='bottom', rotation=90, color='m', weight='bold')
plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.show()

padding=0.0001
rects = plt.bar(np.arange(len(cb_model_score1.get_feature_importance(Pool(X_train, y_train.score1), type='LossFunctionChange'))), 
				cb_model_score1.get_feature_importance(Pool(X_train, y_train.score1), type='LossFunctionChange'), 
				width=0.8, color='y')
for rect in rects:
	height = rect.get_height()
	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 4), 
			 ha='center', va='bottom', rotation=90, color='y', weight='bold')
plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.show()


### Score 2 Feature importance
regr_score2 = RandomForestRegressor(n_estimators=100)
regr_score2.fit(X_train, y_train.score2)

xgb_model_score2 = xgb.XGBRegressor()
xgb_model_score2.fit(X_train, y_train.score2)

cb_model_score2 = CatBoostRegressor()
cb_model_score2.fit(X_train, y_train.score2, eval_set=(X_val, y_val.score2), early_stopping_rounds=10)


padding=0.001
rects = plt.bar(np.arange(len(regr_score2.feature_importances_)), regr_score2.feature_importances_, width=0.8, color='c')
for rect in rects:
	height = rect.get_height()
	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
			 ha='center', va='bottom', rotation=90, color='c', weight='bold')
plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.show()


rects = plt.bar(np.arange(len(xgb_model_score2.feature_importances_)), 
				xgb_model_score2.feature_importances_, 
				width=0.8, color='m')
for rect in rects:
	height = rect.get_height()
	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 3), 
			 ha='center', va='bottom', rotation=90, color='m', weight='bold')
plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.show()

padding=0.0001
rects = plt.bar(np.arange(len(cb_model_score2.get_feature_importance(Pool(X_train, y_train.score2), type='LossFunctionChange'))), 
				cb_model_score2.get_feature_importance(Pool(X_train, y_train.score2), type='LossFunctionChange'), 
				width=0.8, color='y')
for rect in rects:
	height = rect.get_height()
	plt.text(rect.get_x() + rect.get_width() / 2, height + padding, round(height, 4), 
			 ha='center', va='bottom', rotation=90, color='y', weight='bold')
plt.xticks(np.arange(df_train.shape[1]), df_train.columns.values, rotation=45, ha='right')
plt.show()