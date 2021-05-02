import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam


class FootballPoissonModel():

	
	def __init__(self, data):
		self.data = pd.concat([data[['team1', 'team2', 'score1']].assign(home=1).rename(
			columns={'team1': 'team', 'team2': 'opponent', 'score1': 'goals'}),
			data[['team2', 'team1', 'score2']].assign(home=0).rename(
			columns={'team2': 'team', 'team1': 'opponent', 'score2': 'goals'})])


	def fit(self):
		self.model = smf.glm(formula="goals ~ home + team + opponent", data=self.data, 
			family=sm.families.Poisson()).fit()

	@property
	def summary(self):
		return self.model.summary()


	def predict_goals(self, data):
		home_goals_pred = self.model.predict(data.assign(home=1).rename(
						columns={'team1':'team', 'team2':'opponent'}))
		away_goals_pred = self.model.predict(data.assign(home=0).rename(
						columns={'team2':'team', 'team1':'opponent'}))

		return home_goals_pred, away_goals_pred


	def predict_chances(self, data, max_goals=10):

		htg, atg = self.predict_goals(data)
		team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [htg, atg]]



		match_pred = [[np.outer(np.array([i[j] for i in team_pred[0]]), np.array([i[j] for i in team_pred[1]]))] for j in range(0, len(team_pred))]
		print(len(match_pred))


		home_win = np.sum(np.tril(match_pred[1][0], -1))
		draw = np.sum(np.diag(match_pred[1][0]))
		away_win = np.sum(np.triu(match_pred[1][0], 1))

		return home_win, draw, away_win
