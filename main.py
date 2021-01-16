from data_operations import FootballStatsPreparation
# from data_operations import Pipeline
# from models import NeuralNetworkModel
# from stats_plot import StatsPlotter


import pandas as pd

data_url = 'D:\Download\spi_matches.csv'
stats_file_path = 'E:\GitHub\match_predictor_v3\stats.csv'

if __name__ == "__main__":

	

		# Load Five Thirty Eight dataset
		raw_df = pd.read_csv(data_url)
		stats = FootballStatsPreparation(stats_file_path, decay_factor=0.001, num_of_matches=10)
		exclude_features = ['season', 'league_id', 'spi1', 'spi2', 'proj_score1', 'proj_score2', 'prob1', 'prob2', 'probtie', 'importance1', 'importance2']
		stats.add_match_features(raw_df, date='2021-01-16', exclude_features=exclude_features)
		print(stats.data)


	
	# today_games = fdf.get_matchday(date=date)
	# season_games = fdf.get_season(season=season)

	# pipeline = Pipeline()
	# today_games = pipeline.prepare()

	# nn_model = NeuralNetworkModel(name=f'regression_model_21122020')
	# today_predictions = nn_model.predict(today_games)

	# splot = StatsPlotter()
	# splot.plot(today_predictions, season_games)