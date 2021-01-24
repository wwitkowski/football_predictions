from datetime import datetime, timedelta

from data_operations import FootballStats
# from data_operations import Pipeline
# from models import NeuralNetworkModel
# from stats_plot import StatsPlotter
import pandas as pd



if __name__ == "__main__":


		five_days_back = datetime.now() - timedelta(5)
		five_days_back = str(datetime.strftime(five_days_back, '%Y-%m-%d'))

		with FootballStats() as stats:
			if stats.data.empty or stats.last_update_date < five_days_back:
				stats.update()
				print(stats.data)
