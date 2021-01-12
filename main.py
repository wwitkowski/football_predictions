from data_operations import FootballStatsPreparation
# from data_operations import Pipeline
# from models import NeuralNetworkModel
# from stats_plot import StatsPlotter

import os
import pandas as pd

data_url = 'D:\Download\spi_matches.csv'
stats_file_path = 'E:\GitHub\match_predictor_v3\stats.csv'

if __name__ == "__main__":

	if not os.path.isfile(stats_file_path):

		# Load Five Thirty Eight dataset
		raw_df = pd.read_csv(data_url)
		print(raw_df)
		fsp = FootballStatsPreparation(raw_df)
		raw_stats_df = fsp.add_match_stats()
		print(raw_stats_df)

	
	today_games = fdf.get_matchday(date=date)
	season_games = fdf.get_season(season=season)

	pipeline = Pipeline()
	today_games = pipeline.prepare()

	nn_model = NeuralNetworkModel(name=f'regression_model_21122020')
	today_predictions = nn_model.predict(today_games)

	splot = StatsPlotter()
	splot.plot(today_predictions, season_games)