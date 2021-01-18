from datetime import datetime, timedelta

from data_operations import FootballStats
# from data_operations import Pipeline
# from models import NeuralNetworkModel
# from stats_plot import StatsPlotter
import pandas as pd



if __name__ == "__main__":


		yesterday = datetime.now() - timedelta(1)
		yestarday = str(datetime.strftime(yesterday, '%Y-%m-%d'))

		with FootballStats() as stats:
			yestarday_data = stats.one_day(yestarday)[['score1']]
			percent_missing = yestarday_data.isnull().sum() / len(yestarday_data)
			if percent_missing.values == 1:
				stats.update()
