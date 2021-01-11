from data_operations import FootballStatsDataFrame
from data_operations import Pipeline
from models import NeuralNetworkModel
from stats_plot import StatsPlotter

if __name__ == "__main__":

	fdf = FootballStatsDataFrame()
	fdf.update()
	today_games = fdf.get_matchday(date=date)
	season_games = fdf.get_season(season=season)

	pipeline = Pipeline()
	today_games = pipeline.prepare()

	nn_model = NeuralNetworkModel(name=f'regression_model_21122020')
	today_predictions = nn_model.predict(today_games)

	splot = StatsPlotter()
	splot.plot(today_predictions, season_games)