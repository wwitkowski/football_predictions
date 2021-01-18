import pandas as pd
import numpy as np
import math
import os
from datetime import datetime



class BivariatePoisson():


	@staticmethod
	def geometric_function(p, k):
		return ((1-p) ** (k-1)) * p


	@staticmethod
	def biv_poisson(x, y, lambda1, lambda2, lambda3=0.15):

		if lambda1 == 0:
			lambda1 = 0.01
		if lambda2 == 0:
			lambda2 = 0.01

		sum_part = 0
		for i in range(0, min(x, y)+1):
			x_over_i = math.factorial(x) / (math.factorial(i) * math.factorial(x-i))
			y_over_i = math.factorial(y) / (math.factorial(i) * math.factorial(y-i))
			sum_part += x_over_i * y_over_i * math.factorial(i) * ((lambda3 / (lambda1*lambda2))**i)

		f = math.exp(-1 * (lambda1 + lambda2 + lambda3)) * ((lambda1**x) / math.factorial(x)) * ((lambda2**y) / math.factorial(y)) * sum_part
		
		return f


	def poisson_matrix(self, xg1, xg2, shape=(10,10), p=0.01, teta=0.15):

		poisson_matrix = np.zeros(shape)
		for i in range(0, shape[0]):
			for j in range(0, shape[1]):
				if i == j:
					poisson_matrix[j, i] = ((1 - p) * self.biv_poisson(i, j, xg1, xg2)) + (p * self.geometric_function(teta, i))
				else:
					poisson_matrix[j, i] = (1 - p) * self.biv_poisson(i, j, xg1, xg2)

		return poisson_matrix


	def win_probabilities(self, xg1, xg2):

		poisson_matrix = self.poisson_matrix(xg1, xg2)

		homewin_prob = sum(sum(np.triu(poisson_matrix, k=1)))
		awaywin_prob = sum(sum(np.tril(poisson_matrix, k=-1)))
		draw_prob = sum(np.diag(poisson_matrix))

		return homewin_prob, draw_prob, awaywin_prob


	def expected_points(self, xg1, xg2):

		homewin_prob, draw_prob, awaywin_prob = self.win_probabilities(xg1, xg2)


		xpts1 = 3*homewin_prob + draw_prob
		xpts2 = 3*awaywin_prob + draw_prob

		return xpts1, xpts2


class FootballStats:

	file_save_path = 'E:\GitHub\match_predictor_v3\stats.csv'
	source_data_path = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'


	def __init__(self):

		if not os.path.isfile(self.file_save_path):
			self.data = pd.read_csv(self.source_data_path)
			drop_cols = ['prob1', 'prob2', 'probtie', 'proj_score1', 'proj_score2']
			self.data.drop(drop_cols, axis=1, inplace=True)
		else:
			self.data = pd.read_csv(self.file_save_path)
			print('found file')


	@staticmethod
	def result(row1, row2):
		if row1 > row2:
			return 1
		elif row1 == row2:
			return 0
		else:
			return 2


	@staticmethod
	def home_points(row):
		if row == 0:
			return 1
		elif row == 1:
			return 3
		else:
			return 0


	@staticmethod
	def away_points(row):
		if row == 0:
			return 1
		elif row == 2:
			return 3
		else:
			return 0
	

	def new_stats(self, data):	
		# Add nonshot and shot expected goals average
		data['avg_xg1'] = (data['xg1'] + data['nsxg1']) / 2
		data['avg_xg2'] = (data['xg2'] + data['nsxg2']) / 2

		data['adj_avg_xg1'] = (data['xg1'] + data['nsxg1'] + data['adj_score1']) / 3
		data['adj_avg_xg2'] = (data['xg2'] + data['nsxg2'] + data['adj_score2']) / 3

		# Add match result
		data['result'] = data.apply(lambda row: self.result(row.score1, row.score2), axis=1)

		# Add points
		data['pts1'] = data.apply(lambda row: self.home_points(row.result), axis=1).astype('float64')
		data['pts2'] = data.apply(lambda row: self.away_points(row.result), axis=1).astype('float64')

		bp = BivariatePoisson()
		# Add expected points based on expected goals
		data['xPTS1'], data['xPTS2'] = zip(*data.apply(lambda row: bp.expected_points(row.xg1, row.xg2), axis=1))	

		return data


	def one_day(self, date):
		return self.data[self.data.date == date]


	def update(self):
		source_data = pd.read_csv(self.source_data_path)
		update_cols = ['spi1', 'spi2',
					   'score1', 'score2',
					   'importance1', 'importance2',
					   'xg1', 'xg2',
					   'nsxg1', 'nsxg2',
					   'adj_score1', 'adj_score2']

		self.data[update_cols] = source_data[update_cols]
		self.data = self.new_stats(self.data)


	def __enter__(self):
		return self


	def __exit__(self, exc_type, exc_val, exc_tb):
		self.data.to_csv(self.file_save_path, index=False)






class FootballStatsDataFrame():


	def __init__(self, source_data, decay_factor=0.004, num_of_matches=40, null_rate=0.5):


		self.source_data = source_data
		self.save_path = 'E:\GitHub\match_predictor_v3\stats.csv'
		if not os.path.isfile(self.save_path):
			self.data = pd.DataFrame()
		else:
			self.data = pd.read_csv(self.save_path)
		self.decay_factor = decay_factor
		self.num_of_matches = num_of_matches
		self.null_rate = null_rate




	def weights(self, row, date):
		date_to_weight = datetime.strptime(row, '%Y-%m-%d')

		return math.exp(-1 * self.decay_factor * (date - date_to_weight).days)




		# Add expected points based on non-shot expected goals
		#data['ns_xPTS1'], data['ns_xPTS2'] = zip(*data.apply(lambda row: bp.expected_points(row.nsxg1, row.nsxg2), axis=1))

		# Add expected points based on average of non shot and shot expected goals
		#data['avg_xPTS1'], data['avg_xPTS2'] = zip(*data.apply(lambda row: bp.expected_points(row.avg_xg1, row.avg_xg2), axis=1))

		return data


	def get_past_data(self, teams_dict, past_data):

		# Initialize prared stats dataframe
		concat_data = pd.DataFrame()

		# Collect data for all teams
		for team in teams_dict.keys():
			team_data = past_data[((past_data.team1 == team) | (past_data.team2 == team)) & (past_data.league == teams_dict[team])].tail(self.num_of_matches).copy()
			concat_data = pd.concat([concat_data, team_data], ignore_index=True)

		return concat_data


	def add_match_features(self, date, exclude_features=None):

		# Take the past data to calculate the stats
		past_data = self.source_data[self.source_data.date < date].copy()
		past_data.dropna(subset=['score1', 'score2', 'xg1', 'xg2'], inplace=True)

		# Take matchday teams data that is not used for calculating stats
		today_data = self.source_data[self.source_data.date == date]

		# Calculate average stats for home teams
		home_teams_dict = today_data.set_index('team1').to_dict()['league']
		home_teams_past_data = self.get_past_data(home_teams_dict, past_data)

		# Calculate average stats for away teams
		away_teams_dict = today_data.set_index('team2').to_dict()['league']
		away_teams_past_data = self.get_past_data(away_teams_dict, past_data)

		teams_past_data = pd.concat([home_teams_past_data, away_teams_past_data], ignore_index=True)
		teams_past_data.drop_duplicates(inplace=True)

		date = datetime.strptime(date, '%Y-%m-%d')
		teams_past_data['weight'] = teams_past_data.apply(lambda row: self.weights(row.date, date), axis=1)
		weights = teams_past_data[['team1', 'team2']].copy()
		weights['weight'] = teams_past_data.pop('weight')
		weighted_stats = teams_past_data.select_dtypes(include=['float64'])
		weighted_stats = weighted_stats.multiply(weights.weight, axis='index')
		teams_past_data.drop(exclude_features, axis=1, inplace=True)

		# Caluclate averages
		sum_values_home = teams_past_data.groupby(['team1']).sum()
		sum_weights_home = weights.groupby(['team1']).sum()

		sum_values_away = teams_past_data.groupby(['team2']).sum()
		sum_weights_away = weights.groupby(['team2']).sum()

		home_stats_names = [col for col in sum_values_away.columns.values if '1' in col]
		away_stats_names = [col for col in sum_values_away.columns.values if '2' in col]
		rename_away_stats_dict = dict(zip(home_stats_names + away_stats_names, away_stats_names + home_stats_names))

		sum_values_away = sum_values_away.rename(columns=rename_away_stats_dict)
		sum_values = sum_values_home.add(sum_values_away, fill_value=0)
		sum_weights = sum_weights_home.add(sum_weights_away, fill_value=0)
		avg_values = sum_values.div(sum_weights.weight, axis=0)

		today_data = today_data.merge(avg_values, left_on='team1', right_index=True, how='inner', suffixes=('', '_home'))
		today_data = today_data.merge(avg_values, left_on='team2', right_index=True, how='inner', suffixes=('', '_away'))

		self.data = pd.concat([self.data, today_data], sort=False, ignore_index=True)


	def __exit__(self):
		self.data.to_csv(self.save_path)


