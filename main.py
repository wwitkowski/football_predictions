from datetime import datetime, timedelta
from data_operations import FootballStats
# from data_operations import Pipeline
# from models import NeuralNetworkModel
# from stats_plot import StatsPlotter
import pandas as pd


if __name__ == "__main__":

		three_days_back = datetime.now() - timedelta(3)
		three_days_back = str(datetime.strftime(three_days_back, '%Y-%m-%d'))

		today = datetime.now().strftime('%Y-%m-%d')

		with FootballStats() as stats:
			if stats.data.empty or stats.last_update_date < three_days_back:
				stats.update()
			today_matchday = stats.one_day(today)
			print(today_matchday)
