from datetime import datetime, timedelta
from data_operations import Dataset



source1 = ['https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv']

leagues = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'F1', 'N1', 'SP1', 'P1']
seasons = [1617, 1718, 1819, 1920, 2021]
source2_link = 'http://www.football-data.co.uk/mmz4281'

source2 = []
for league in leagues:
	for season in seasons:
		source2.append(f'{source2_link}/{season}/{league}.csv')

ds1 = Dataset(source=source1, save_folder='FiveThirtyEight', date_column='date')

ds2_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 
			'shots1', 'shots2', 'shotsot1', 'shotsot2', 
			'fouls1', 'fouls2', 'corners1', 'corners2', 
			'yellow1', 'yellow2', 'red1', 'red2', 
			'MaxH', 'MaxD','MaxA', 'AvgH', 'AvgD', 'AvgA']

ds2_rename_dict = {'BbMxH': 'MaxH', 'BbMxD': 'MaxD', 
				'BbMxA': 'MaxA', 'BbAvH': 'AvgH', 
				'BbAvD': 'AvgD', 'BbAvA': 'AvgA',
				'HS': 'shots1', 'AS': 'shots2', 
				'HST': 'shotsot1', 'AST': 'shotsot2', 
				'HF': 'fouls1', 'AF': 'fouls2', 
				'HC': 'corners1', 'AC': 'corners2', 
				'HY': 'yellow1', 'AY': 'yellow2', 
				'HR': 'red1', 'AR': 'red2'}

ds2 = Dataset(source=source2, save_folder='Football-data/data/', 
			columns=ds2_columns, date_column='Date',
			rename_dict=ds2_rename_dict, sub_folder=True)


today = datetime.now().strftime('%Y-%m-%d')
two_days_back = datetime.now() - timedelta(2)
two_days_back = two_days_back.strftime('%Y-%m-%d')

if two_days_back > min(ds1.last_valid_date, ds2.last_valid_date):
	print('Updating...')
	ds1.update()
	ds2.update()

l_keys = ['date', 'team1', 'team2']
r_keys = ['date', 'HomeTeam', 'AwayTeam']






