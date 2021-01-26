import pandas as pd
import numpy as np
import math
import os
from datetime import datetime


class FootballStats:
	"""
	Class for handling download data from FiveThirtyEight and Football-data dataset and joining
	them into one pseudo api that allows to easily retrieve matches from specdific matchday (date).
	"""

	source1 = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'
	leagues = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'F1', 'N1', 'SP1', 'P1']
	source2 = 'http://www.football-data.co.uk/mmz4281/2021/'
	final_df_save_path = 'E:\GitHub\match_predictor_v3\stats.csv'

	def __init__(self):
		self.data = pd.DataFrame()


	def one_day(self, date):
		"""
		Function returning matches for a specific matchday.
		"""
		return self.data[self.data.date == date]


	@property
	def last_update_date(self):
		"""
		Returns last update date. The date is found where the last not null value is.
		"""
		last_valid_idxs = self.data.apply(pd.Series.last_valid_index)
		last_update_idx = min(last_valid_idxs.values)

		return self.data.date.iloc[last_update_idx]

	
	def update(self):
		"""
		Function to update the datasets with altest data.
		"""

		# Source 1 - FiveThirtyEight
		source1_df = pd.read_csv(self.source1)

		# Source 2 - Football-data
		source2_df = pd.DataFrame()

		# Pick relevant columns with used features from source 2
		columns = ['Date', 'HomeTeam', 'AwayTeam', 
				'HS', 'AS', 'HST', 'AST', 
				'HF', 'AF', 'HC', 'AC', 'HY', 
				'AY', 'HR', 'AR', 'MaxH', 'MaxD',
				'MaxA', 'AvgH', 'AvgD', 'AvgA']

		# Load data from past season from hard drive
		rootdir = 'Football-data\data'
		for subdir, dirs, files in os.walk(rootdir):
			for file in files:
				data = pd.read_csv(os.path.join(subdir, file))
				try:
					data['Date'] = pd.to_datetime(data.Date, format='%d/%m/%y')
				except ValueError:
					data['Date'] = pd.to_datetime(data.Date, format='%d/%m/%Y')
				try:
					data = data[columns]
				except KeyError:
					rename_dict = {'BbMxH': 'MaxH', 'BbMxD': 'MaxD', 'BbMxA': 'MaxA', 'BbAvH': 'AvgH', 'BbAvD': 'AvgD', 'BbAvA': 'AvgA'}
					data.rename(columns=rename_dict, inplace=True)
					data = data[columns]

				source2_df = pd.concat([source2_df, data], ignore_index=True)

		# Load data the freshest data from internet for current season
		for league in self.leagues:
			link = f'{self.source2}/{league}.csv'
			data = pd.read_csv(link)
			data['Date'] = pd.to_datetime(data.Date, format='%d/%m/%Y')
			data = data[columns]
			source2_df = pd.concat([source2_df, data], ignore_index=True)

		# Load mapping
		df_mapping = pd.read_csv('E:\GitHub\match_predictor_v3\Football-data\mapping.csv')
		mapping = df_mapping.set_index('replace').to_dict()['replace_with']

		# Adapt team names using mapping
		source2_df.replace(mapping, inplace=True)
		
		# Conver date to date format
		source1_df['date'] = pd.to_datetime(source1_df.date)

		# Merge predictions data with odds data
		source1_df.reset_index(drop=True)
		left_keys = ['date', 'team1', 'team2']
		right_keys = ['Date', 'HomeTeam', 'AwayTeam']
		self.data = pd.merge(source1_df, source2_df, how='left', left_on=left_keys, right_on=right_keys)
		self.data.drop(right_keys, axis=1, inplace=True)

		self.data['date'] = self.data.date.apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))

		return self.data


	def __enter__(self):
		try:
			self.data = pd.read_csv(self.final_df_save_path)
		except FileNotFoundError:
			self.update()

		return self


	def __exit__(self, exc_type, exc_val, exc_tb):
		self.data.to_csv(self.final_df_save_path, index=False)



# 	class BivariatePoisson():


# 	@staticmethod
# 	def geometric_function(p, k):
# 		return ((1-p) ** (k-1)) * p


# 	@staticmethod
# 	def biv_poisson(x, y, lambda1, lambda2, lambda3=0.15):

# 		if lambda1 == 0:
# 			lambda1 = 0.01
# 		if lambda2 == 0:
# 			lambda2 = 0.01

# 		sum_part = 0
# 		for i in range(0, min(x, y)+1):
# 			x_over_i = math.factorial(x) / (math.factorial(i) * math.factorial(x-i))
# 			y_over_i = math.factorial(y) / (math.factorial(i) * math.factorial(y-i))
# 			sum_part += x_over_i * y_over_i * math.factorial(i) * ((lambda3 / (lambda1*lambda2))**i)

# 		f = math.exp(-1 * (lambda1 + lambda2 + lambda3)) * ((lambda1**x) / math.factorial(x)) * ((lambda2**y) / math.factorial(y)) * sum_part
		
# 		return f


# 	def poisson_matrix(self, xg1, xg2, shape=(10,10), p=0.01, teta=0.15):

# 		poisson_matrix = np.zeros(shape)
# 		for i in range(0, shape[0]):
# 			for j in range(0, shape[1]):
# 				if i == j:
# 					poisson_matrix[j, i] = ((1 - p) * self.biv_poisson(i, j, xg1, xg2)) + (p * self.geometric_function(teta, i))
# 				else:
# 					poisson_matrix[j, i] = (1 - p) * self.biv_poisson(i, j, xg1, xg2)

# 		return poisson_matrix


# 	def win_probabilities(self, xg1, xg2):

# 		poisson_matrix = self.poisson_matrix(xg1, xg2)

# 		homewin_prob = sum(sum(np.triu(poisson_matrix, k=1)))
# 		awaywin_prob = sum(sum(np.tril(poisson_matrix, k=-1)))
# 		draw_prob = sum(np.diag(poisson_matrix))

# 		return homewin_prob, draw_prob, awaywin_prob


# 	def expected_points(self, xg1, xg2):

# 		homewin_prob, draw_prob, awaywin_prob = self.win_probabilities(xg1, xg2)


# 		xpts1 = 3*homewin_prob + draw_prob
# 		xpts2 = 3*awaywin_prob + draw_prob

# 		return xpts1, xpts2


# 	@staticmethod
# 	def result(row1, row2):
# 		if row1 > row2:
# 			return 1
# 		elif row1 == row2:
# 			return 0
# 		else:
# 			return 2


# 	@staticmethod
# 	def home_points(row):
# 		if row == 0:
# 			return 1
# 		elif row == 1:
# 			return 3
# 		else:
# 			return 0


# 	@staticmethod
# 	def away_points(row):
# 		if row == 0:
# 			return 1
# 		elif row == 2:
# 			return 3
# 		else:
# 			return 0


# 	def new_stats(self, data):	
# 		# Add nonshot and shot expected goals average
# 		data['avg_xg1'] = (data['xg1'] + data['nsxg1']) / 2
# 		data['avg_xg2'] = (data['xg2'] + data['nsxg2']) / 2

# 		data['adj_avg_xg1'] = (data['xg1'] + data['nsxg1'] + data['adj_score1']) / 3
# 		data['adj_avg_xg2'] = (data['xg2'] + data['nsxg2'] + data['adj_score2']) / 3

# 		# Add match result
# 		data['result'] = data.apply(lambda row: self.result(row.score1, row.score2), axis=1)

# 		# Add points
# 		data['pts1'] = data.apply(lambda row: self.home_points(row.result), axis=1).astype('float64')
# 		data['pts2'] = data.apply(lambda row: self.away_points(row.result), axis=1).astype('float64')

# 		bp = BivariatePoisson()
# 		# Add expected points based on expected goals
# 		data['xPTS1'], data['xPTS2'] = zip(*data.apply(lambda row: bp.expected_points(row.xg1, row.xg2), axis=1))	

# 		return data

# class FootballStatsDataFrame():


# 	def __init__(self, source_data, decay_factor=0.004, num_of_matches=40, null_rate=0.5):


# 		self.source_data = source_data
# 		self.save_path = 'E:\GitHub\match_predictor_v3\stats.csv'
# 		if not os.path.isfile(self.save_path):
# 			self.data = pd.DataFrame()
# 		else:
# 			self.data = pd.read_csv(self.save_path)
# 		self.decay_factor = decay_factor
# 		self.num_of_matches = num_of_matches
# 		self.null_rate = null_rate




# 	def weights(self, row, date):
# 		date_to_weight = datetime.strptime(row, '%Y-%m-%d')

# 		return math.exp(-1 * self.decay_factor * (date - date_to_weight).days)



# 	def get_past_data(self, teams_dict, past_data):

# 		# Initialize prared stats dataframe
# 		concat_data = pd.DataFrame()

# 		# Collect data for all teams
# 		for team in teams_dict.keys():
# 			team_data = past_data[((past_data.team1 == team) | (past_data.team2 == team)) & (past_data.league == teams_dict[team])].tail(self.num_of_matches).copy()
# 			concat_data = pd.concat([concat_data, team_data], ignore_index=True)

# 		return concat_data


# 	def add_match_features(self, date, exclude_features=None):

# 		# Take the past data to calculate the stats
# 		past_data = self.source_data[self.source_data.date < date].copy()
# 		past_data.dropna(subset=['score1', 'score2', 'xg1', 'xg2'], inplace=True)

# 		# Take matchday teams data that is not used for calculating stats
# 		today_data = self.source_data[self.source_data.date == date]

# 		# Calculate average stats for home teams
# 		home_teams_dict = today_data.set_index('team1').to_dict()['league']
# 		home_teams_past_data = self.get_past_data(home_teams_dict, past_data)

# 		# Calculate average stats for away teams
# 		away_teams_dict = today_data.set_index('team2').to_dict()['league']
# 		away_teams_past_data = self.get_past_data(away_teams_dict, past_data)

# 		teams_past_data = pd.concat([home_teams_past_data, away_teams_past_data], ignore_index=True)
# 		teams_past_data.drop_duplicates(inplace=True)

# 		date = datetime.strptime(date, '%Y-%m-%d')
# 		teams_past_data['weight'] = teams_past_data.apply(lambda row: self.weights(row.date, date), axis=1)
# 		weights = teams_past_data[['team1', 'team2']].copy()
# 		weights['weight'] = teams_past_data.pop('weight')
# 		weighted_stats = teams_past_data.select_dtypes(include=['float64'])
# 		weighted_stats = weighted_stats.multiply(weights.weight, axis='index')
# 		teams_past_data.drop(exclude_features, axis=1, inplace=True)

# 		# Caluclate averages
# 		sum_values_home = teams_past_data.groupby(['team1']).sum()
# 		sum_weights_home = weights.groupby(['team1']).sum()

# 		sum_values_away = teams_past_data.groupby(['team2']).sum()
# 		sum_weights_away = weights.groupby(['team2']).sum()

# 		home_stats_names = [col for col in sum_values_away.columns.values if '1' in col]
# 		away_stats_names = [col for col in sum_values_away.columns.values if '2' in col]
# 		rename_away_stats_dict = dict(zip(home_stats_names + away_stats_names, away_stats_names + home_stats_names))

# 		sum_values_away = sum_values_away.rename(columns=rename_away_stats_dict)
# 		sum_values = sum_values_home.add(sum_values_away, fill_value=0)
# 		sum_weights = sum_weights_home.add(sum_weights_away, fill_value=0)
# 		avg_values = sum_values.div(sum_weights.weight, axis=0)

# 		today_data = today_data.merge(avg_values, left_on='team1', right_index=True, how='inner', suffixes=('', '_home'))
# 		today_data = today_data.merge(avg_values, left_on='team2', right_index=True, how='inner', suffixes=('', '_away'))

# 		self.data = pd.concat([self.data, today_data], sort=False, ignore_index=True)

