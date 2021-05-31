import pandas as pd
import numpy as np
import math
import os
import re
from datetime import datetime, timedelta
from sources import Source1, Source2
from models import FootballPoissonModel
from sklearn.neighbors import NearestNeighbors


class DateHandler:
	"""
	Class for handling the different dates.
	"""

	def __init__(self):
		pass

	@staticmethod
	def to_date(date_string):
		"""
		Function to convert date formats to datetime object.
		"""
		splitter = re.search("[./-]", date_string)[0]
		groups = re.split("([./-])", date_string)
		date_len = len(date_string)
		if len(groups[0]) == 2 and date_len == 8:
			return datetime.strptime(date_string, f'%d{splitter}%m{splitter}%y')
		elif len(groups[0]) == 4 and date_len == 10:
			return datetime.strptime(date_string, f'%Y{splitter}%m{splitter}%d')
		else:
			return datetime.strptime(date_string, f'%d{splitter}%m{splitter}%Y')



class DatasetLoader:
	"""
	Class for downlaoding data from given source links and saving them on hard drive.
	"""

	def __init__(self, save_folder):
		self.save_folder = save_folder

	
	def load(self, link, sub_folder=False):
		"""
		Function loading all files either from hard drive - if they exist, or downloading from source links.
		"""
		file_name = link.split('/')[-1]
		if sub_folder:
			sub_folder = link.split('/')[-2]
			path = f'{self.save_folder}/{sub_folder}/{file_name}'
		else:
			path = f'{self.save_folder}/{file_name}'
		try:
			data = pd.read_csv(path)
			return data	
		except FileNotFoundError:
			data = pd.read_csv(link)
			data.to_csv(path, index=False)
			return data		


class Dataset(DatasetLoader):
	"""
	Class for handling csv datasets downloaded from internet.
	"""

	def __init__(self, save_folder, source, date_column, columns=None, rename_dict=None, sub_folder=False):
		super().__init__(save_folder)
		self.data = pd.DataFrame()
		self.source = source
		self.columns = columns
		self.rename_dict = rename_dict
		self.sub_folder = sub_folder
		self.date_column = date_column

		self.concat_dataframes()
		self.handle_dates()


	def handle_dates(self):
		"""
		Function renaming date column to uniform name and formatting to uniform format YYYY-MM-DD.
		"""
		self.data.rename(columns={self.date_column: 'date'}, inplace=True)
		self.data['date'] = self.data['date'].apply(lambda x: DateHandler.to_date(x).strftime('%Y-%m-%d'))
	

	def concat_dataframes(self):
		"""
		Function concatenating all dataframes from the source link.
		"""
		for link in self.source:
			try:
				temp_df = self.load(link, self.sub_folder)
				if self.columns is not None:
					try:
						temp_df = temp_df[self.columns]
					except KeyError:
						temp_df.rename(columns=self.rename_dict, inplace=True)
						try:
							temp_df = temp_df[self.columns]
						except KeyError as e:
							print(f'{link}: {e} - data skipped.')
							continue
				self.data = pd.concat([self.data, temp_df], sort=False, ignore_index=True)
			except Exception as e:
				print(e)


	def update(self):
		"""
		Function updating the dataset bby downloading the most recent files again.
		"""
		folders = [folder[0] for folder in os.walk(self.save_folder)]
		most_recent = max(folders)
		for file in os.listdir(most_recent):
			os.remove(f'{most_recent}/{file}')
		self.data = pd.DataFrame()
		self.concat_dataframes()
		self.handle_dates()


	@property
	def last_valid_date(self):
		"""
		Returns last update date. The date is found where the last not null value is.
		"""
		last_valid_idxs = self.data.apply(pd.Series.last_valid_index)
		last_update_idx = min(last_valid_idxs.values)

		return self.data.date.iloc[last_update_idx]


# class PoissonInflatedModel:
# 	"""
# 	Poisson Inflated Model for calculating Match outcome probabilities and expected points.
# 	Model Inflates the draw outcome 
# 	------------------------
# 	parameters:
# 	p - determines the inflation strenght
# 	teta - determines how much the model is inflated towards 0-0 match outcome
# 	shape - matrix shape. (10, 10) means that probabilities will be calculated for up to 10-10 score
# 	"""

# 	def __init__(self, p=0.1, teta=0.15, shape=(10, 10)):
# 		self.p = p
# 		self.teta = teta
# 		self.shape = shape


# 	@staticmethod
# 	def geometric_function(p, k):
# 		return ((1-p) ** (k-1)) * p


# 	@staticmethod
# 	def poisson(mu, k):
# 		f = ((mu ** k)*(math.exp(1)**(-mu))) / math.factorial(k)
# 		return f


# 	def poisson_matrix(self, xg1, xg2):
# 		poisson_matrix = np.zeros(self.shape)
# 		for i in range(0, self.shape[0]):
# 			for j in range(0, self.shape[1]):
# 				if i == j:
# 					poisson_matrix[j, i] = (1 - self.p) * self.poisson(xg1, i) * self.poisson(xg2, j) + self.p * self.geometric_function(self.teta, i)
# 				else:
# 					poisson_matrix[j, i] = (1 - self.p) * self.poisson(xg1, i) * self.poisson(xg2, j)

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


class FootballStats:
	mapping_path = 'Football-data\mapping.csv'
	path = 'stats.csv'


	def __init__(self):
		self.ds1 = Dataset(source=Source1.links, 
					  save_folder=Source1.save_folder,
					  date_column=Source1.date_column)

		self.ds2 = Dataset(source=Source2.links,
					  save_folder=Source2.save_folder,
					  date_column=Source2.date_column,
					  columns=Source2.columns,
					  rename_dict=Source2.rename_dict,
					  sub_folder=True)

		today = datetime.now().strftime('%Y-%m-%d')
		lookback_date = datetime.now() - timedelta(3)
		lookback_date = lookback_date.strftime('%Y-%m-%d')
		print(lookback_date)
		if lookback_date > min(self.ds1.last_valid_date, self.ds2.last_valid_date):
			print('update')
			try:
				os.remove(self.path)
			except FileNotFoundError:
				pass


	def load(self):
		self.ds1.update()
		self.ds2.update()

		df_mapping = pd.read_csv(self.mapping_path)
		mapping = df_mapping.set_index('replace').to_dict()['replace_with']

		# Adapt team names using mapping
		self.ds2.data.replace(mapping, inplace=True)
		lkeys = ['date', 'team1', 'team2']
		rkeys = ['date', 'HomeTeam', 'AwayTeam']
		self.data = pd.merge(self.ds1.data, self.ds2.data, how='left', left_on=lkeys, right_on=rkeys)
		self.add_match_stats()


	@staticmethod
	def points(row):
		if row == 'D':
			return 1, 1
		elif row == 'H':
			return 3, 0
		else:
			return 0, 3


	def add_match_stats(self):	
		# Add nonshot and shot expected goals average
		self.data['avg_xg1'] = (self.data['xg1'] + self.data['nsxg1']) / 2
		self.data['avg_xg2'] = (self.data['xg2'] + self.data['nsxg2']) / 2

		self.data['adj_avg_xg1'] = (self.data['xg1'] + self.data['nsxg1'] + self.data['adj_score1']) / 3
		self.data['adj_avg_xg2'] = (self.data['xg2'] + self.data['nsxg2'] + self.data['adj_score2']) / 3

		# Add points
		self.data['pts1'], self.data['pts2'] = zip(*self.data.apply(lambda row: self.points(row.FTR), axis=1))

		# Add expected points based on expected goals
		p = FootballPoissonModel()
		self.data[['xg1', 'xg2']] = self.data[['xg1', 'xg2']].fillna(value=0)
		h, d, a = p.predict_chances(self.data.xg1.values, self.data.xg2.values)
		self.data['xwin1'], self.data['xdraw'], self.data['xwin2'] = h, d, a
		self.data['xpts1'] = 3 * self.data['xwin1'] + self.data['xdraw']
		self.data['xpts2'] = 3 * self.data['xwin2'] + self.data['xdraw']

		self.data['xgshot1'] = self.data['xg1'] / self.data['shots1']
		self.data.xgshot1.replace([np.inf, -np.inf], 0, inplace=True)
		self.data['xgshot2'] = self.data['xg2'] / self.data['shots2']
		self.data.xgshot2.replace([np.inf, -np.inf], 0, inplace=True)

		self.data['convrate1'] = self.data['score1'] / self.data['shots1']
		self.data.convrate1.fillna(0, inplace=True)
		self.data['convrate2'] = self.data['score2'] / self.data['shots2']
		self.data.convrate2.fillna(0, inplace=True)

		self.data['cards1'] = self.data['yellow1'] + 2 * self.data['red1']
		self.data['cards2'] = self.data['yellow2'] + 2 * self.data['red2']


	def __enter__(self):
		try:
			self.data = pd.read_csv(self.path)
		except FileNotFoundError:
			self.load()
		
		return self


	def __exit__(self, exc_type, exc_val, exc_tb):
		self.data.to_csv(self.path, index=False)


class TeamStats:

	def __init__(self, decay_factor, num_of_matches, min_matches):
		self.decay_factor = decay_factor
		self.num_of_matches = num_of_matches
		self.min_matches = min_matches


	@staticmethod
	def stats_from_similar(present_df, past_df, by, n):
		search_array = np.array(past_df[by])
		input_array = np.array(present_df[by])
		neigh = NearestNeighbors(n_neighbors=n)
		neigh.fit(search_array)
		
		_, indices = neigh.kneighbors(input_array)
		
		return indices


	def weights(self, row, date):
		date_to_weight = datetime.strptime(row, '%Y-%m-%d')

		return math.exp(-1 * self.decay_factor * (date - date_to_weight).days)


	def get_past_data(self, teams_dict, past_data, past_matches_num):

		# Initialize prared stats dataframe
		concat_data = pd.DataFrame()

		# Collect data for all teams
		for team in teams_dict.keys():
			team_data = past_data[((past_data.team1 == team) | (past_data.team2 == team)) & (past_data.league == teams_dict[team])].tail(past_matches_num).copy()
			if team_data.shape[0] <= self.min_matches:
				continue
			concat_data = pd.concat([concat_data, team_data], ignore_index=True)

		return concat_data


	def get_past_average(self, df, date, exclude_features=[]):

		# Take the past data to calculate the stats
		past_data = df[df.date < date].copy()
		past_data.dropna(subset=['score1', 'score2', 'xg1', 'xg2', 'nsxg1', 'nsxg2', 'HomeTeam', 'AwayTeam', 'shots1', 'shots2'], inplace=True)

		league_avgs = past_data[['league_id', 'xg1', 'xg2', 'score1', 'score2']].groupby(['league_id']).mean()
		
		# Take matchday teams data that is not used for calculating stats
		today_data = df[df.date == date]

		similar_idxs = self.stats_from_similar(today_data, past_data, by=['spi1', 'spi2'], n=100)

		similar_df = pd.DataFrame()
		for num, idx in enumerate(similar_idxs):
			target = today_data[['team1', 'team2']].iloc[num]
			result = past_data[['score1', 'score2', 'xg1', 'xg2']].iloc[idx].mean()
			wins_perc = past_data['FTR'].iloc[idx].value_counts(normalize=True)
			final = pd.concat([target, result])
			final = pd.concat([final, wins_perc])
			similar_df = similar_df.append(final, ignore_index=True)

 
		# Calculate average stats for home teams
		home_teams_dict = today_data.set_index('team1').to_dict()['league']
		home_teams_past_data = self.get_past_data(home_teams_dict, past_data, self.num_of_matches)
		home_teams_past_luck = self.get_past_data(home_teams_dict, past_data, 10)

		# Calculate average stats for away teams
		away_teams_dict = today_data.set_index('team2').to_dict()['league']
		away_teams_past_data = self.get_past_data(away_teams_dict, past_data, self.num_of_matches)
		away_teams_past_luck = self.get_past_data(away_teams_dict, past_data, 10)

		teams_past_data = pd.concat([home_teams_past_data, away_teams_past_data], ignore_index=True)
		teams_past_data.drop_duplicates(inplace=True)


		date = datetime.strptime(date, '%Y-%m-%d')
		teams_past_data['weight'] = teams_past_data.apply(lambda row: self.weights(row.date, date), axis=1)
		teams_past_data.drop(exclude_features, axis=1, inplace=True)
		
		# Caluclate averages
		home_stats_names = [col for col in teams_past_data.columns.values if '1' in col]
		away_stats_names = [col for col in teams_past_data.columns.values if '2' in col]
		rename_away_stats_dict = dict(zip(home_stats_names + away_stats_names, away_stats_names + home_stats_names))

		avg_weight_stats = pd.concat([teams_past_data, teams_past_data.rename(
						 			 columns=rename_away_stats_dict)], ignore_index=True, sort=False)

		weighted_stats = teams_past_data.select_dtypes(include=['float64']).columns.values

		avg_weight_stats = avg_weight_stats.groupby(['team1']).apply(lambda x: pd.Series(np.average(x[weighted_stats], weights=x.weight, axis=0), weighted_stats))


		teams_past_luck = pd.concat([home_teams_past_luck, away_teams_past_luck], ignore_index=True)
		teams_past_luck.drop_duplicates(inplace=True)

		teams_past_luck['home_team_luck'] = teams_past_luck['score1'] - teams_past_luck['xg1']
		teams_past_luck['away_team_luck'] = teams_past_luck['score2'] - teams_past_luck['xg2']

		avg_luck = pd.concat([teams_past_luck[['team1', 'home_team_luck']].rename(
						 	 												columns={'home_team_luck': 'team_luck'}), 
							  teams_past_luck[['team2', 'away_team_luck']].rename(
						 	 												columns={'team2': 'team1',
						 	 														 'away_team_luck': 'team_luck'})], 
							  												ignore_index=True)

		avg_luck = avg_luck.groupby(['team1']).mean()

		today_data = today_data.merge(avg_weight_stats, left_on='team1', right_index=True, how='inner', suffixes=('', '_home'))
		today_data = today_data.merge(avg_weight_stats, left_on='team2', right_index=True, how='inner', suffixes=('', '_away'))
		today_data = today_data.merge(avg_luck, left_on='team1', right_index=True, how='inner')
		today_data = today_data.merge(avg_luck, left_on='team2', right_index=True, how='inner', suffixes=('_home', '_away'))
		today_data = today_data.merge(league_avgs, on='league_id', right_index=True, how='inner', suffixes=('', '_league'))
		today_data = today_data.merge(similar_df, on=['team1', 'team2'], right_index=True, how='inner', suffixes=('', '_similar'))

		return today_data


class Bookmaker():

	def __init__(self, data, stake, odds='max', kelly=False):
		self.data = data
		self.stake = stake
		self.odds = odds
		self.kelly = kelly

	def _bet(self, row):
		if self.odds == 'max':
			if row.MaxH > 1/row.homewin_pred:
				if row.MaxD > 1/row.draw_pred:
					return 'HD'
				elif row.MaxA > 1/row.awaywin_pred:
					return 'HA'
				else:
					return 'H'
			elif row.MaxA > 1/row.homewin_pred:
				if row.MaxD > 1/row.draw_pred:
					return 'AD'
				else:
					return 'A'
			elif row.MaxD > 1/row.draw_pred:
				return 'D'
			else:
				return 'X'
		if self.odds == 'avg':
			if row.AvgH > 1/row.homewin_pred:
				if row.AvgD > 1/row.draw_pred:
					return 'HD'
				elif row.AvgA > 1/row.awaywin_pred:
					return 'HA'
				else:
					return 'H'
			elif row.AvgA > 1/row.homewin_pred:
				if row.AvgD > 1/row.draw_pred:
					return 'AD'
				else:
					return 'A'
			elif row.AvgD > 1/row.draw_pred:
				return 'D'
			else:
				return 'X'


	def _prediction_odds(self, row):
		if self.odds == 'max':
			if row.BET == 'H':
				return row.MaxH
			elif row.BET == 'HD':
				return 1/(1/row.MaxH + 1/row.MaxD)
			elif row.BET == 'HA':
				return 1/(1/row.MaxH + 1/row.MaxA)
			elif row.BET == 'D':
				return row.MaxD
			elif row.BET == 'AD':
				return 1/(1/row.MaxD + 1/row.MaxA)
			elif row.BET == 'A':
				return row.MaxA
			else:
				return 1
		if self.odds == 'avg':
			if row.BET == 'H':
				return row.AvgH
			elif row.BET == 'HD':
				return 1/(1/row.AvgH + 1/row.AvgD)
			elif row.BET == 'HA':
				return 1/(1/row.AvgH + 1/row.AvgA)
			elif row.BET == 'D':
				return row.AvgD
			elif row.BET == 'AD':
				return 1/(1/row.AvgD + 1/row.AvgA)
			elif row.BET == 'A':
				return row.AvgA
			else:
				return 1

	def _return(self, row):
		if row.FTR in row.BET:
			return (self.stake/row.prediction_odds)*row.prediction_odds - (self.stake/row.prediction_odds)
		elif row.BET == 'X':
			return 0
		else:
			return -(self.stake/row.prediction_odds)

	def _over_under(self, row):
		if row.score1 + row.score2 > 2.5:
			return 'Over'
		else:
			return 'Under'		


	def _bet_over(self, row):
		if self.odds == 'max':
			if row['Max>2.5'] > 1/row['>2.5_pred']:
				return 'Over'
			elif row['Max<2.5'] > 1/row['<2.5_pred']:
				return 'Under'
			else:
				return 'X'
		if self.odds == 'avg':
			if row['Avg>2.5'] > 1/row['>2.5_pred']:
				return 'Over'
			elif row['Avg<2.5'] > 1/row['<2.5_pred']:
				return 'Under'
			else:
				return 'X'


	def _prediction_odds_over(self, row):
		if self.odds == 'max':
			if row.BET_OVER_UNDER == 'Over':
				return row['Max>2.5']
			elif row.BET_OVER_UNDER == 'Under':
				return row['Max<2.5']
			else:
				return 1
		if self.odds == 'avg':
			if row.BET_OVER_UNDER == 'Over':
				return row['Max>2.5']
			elif row.BET_OVER_UNDER == 'Under':
				return row['Max<2.5']
			else:
				return 1

	def _return_over(self, row):
		if row.OVER_UNDER in row.BET_OVER_UNDER:
			return (self.stake/row.prediction_odds_over)*row.prediction_odds_over - (self.stake/row.prediction_odds_over)
		elif row.BET_OVER_UNDER == 'X':
			return 0
		else:
			return -(self.stake/row.prediction_odds_over)


	def calculate(self):
		self.data['BET'] = self.data.apply(lambda row: self._bet(row), axis=1)
		self.data['prediction_odds'] = self.data.apply(lambda row: self._prediction_odds(row), axis=1)
		self.data['bet_return'] = self.data.apply(lambda row: self._return(row), axis=1)

		self.data['OVER_UNDER'] = self.data.apply(lambda row: self._over_under(row), axis=1)
		self.data['BET_OVER_UNDER'] = self.data.apply(lambda row: self._bet_over(row), axis=1)
		self.data['prediction_odds_over'] = self.data.apply(lambda row: self._prediction_odds_over(row), axis=1)
		self.data['bet_return_over'] = self.data.apply(lambda row: self._return_over(row), axis=1)






