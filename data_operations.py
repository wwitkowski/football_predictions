import pandas as pd
import numpy as np
import math


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



class FootballStatsPreparation():


	def __init__(self, df, decay_factor=0.004, num_of_matches=40, null_rate=0.5):

		self.df = df
		self.decay_factor = decay_factor
		self.num_of_matches = num_of_matches
		self.null_rate = null_rate


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


	def add_match_stats(self):
		
		# Add nonshot and shot expected goals average
		self.df['avg_xg1'] = (self.df['xg1'] + self.df['nsxg1']) / 2
		self.df['avg_xg2'] = (self.df['xg2'] + self.df['nsxg2']) / 2

		self.df['adj_avg_xg1'] = (self.df['xg1'] + self.df['nsxg1'] + self.df['adj_score1']) / 3
		self.df['adj_avg_xg2'] = (self.df['xg2'] + self.df['nsxg2'] + self.df['adj_score2']) / 3

		# Add match result
		self.df['result'] = self.df.apply(lambda row: self.result(row.score1, row.score2), axis=1)

		# Add points
		self.df['pts1'] = self.df.apply(lambda row: self.home_points(row.result), axis=1).astype('float64')
		self.df['pts2'] = self.df.apply(lambda row: self.away_points(row.result), axis=1).astype('float64')

		bp = BivariatePoisson()
		# Add expected points based on expected goals
		self.df['xPTS1'], self.df['xPTS2'] = zip(*self.df.apply(lambda row: bp.expected_points(row.xg1, row.xg2), axis=1))

		# Add expected points based on non-shot expected goals
		self.df['ns_xPTS1'], self.df['ns_xPTS2'] = zip(*self.df.apply(lambda row: bp.expected_points(row.nsxg1, row.nsxg2), axis=1))

		# Add expected points based on average of non shot and shot expected goals
		self.df['avg_xPTS1'], self.df['avg_xPTS2'] = zip(*self.df.apply(lambda row: bp.expected_points(row.avg_xg1, row.avg_xg2), axis=1))

		return self.df


	def add_match_features(end_date, start_date=None):
		pass









