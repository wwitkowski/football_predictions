from data_operations import FootballStats
from data_operations import TeamStats
from datetime import datetime, timedelta

DATE = datetime.now().strftime('%Y-%m-%d')
EXCLUDE_FEATURES = ['prob1','prob2',
					'probtie','proj_score1',
					'proj_score2','MaxH', 
					'MaxD','MaxA',
					'AvgH','AvgD',
					'AvgA', 'league_id',
					'season']


ts = TeamStats(decay_factor=0, num_of_matches=4)
with FootballStats() as stats:
	today_data = ts.get_past_average(stats.data, date=DATE, exclude_features=EXCLUDE_FEATURES)
	print(today_data)

