class Source1:
	links = ['https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv']
	save_folder = 'FiveThirtyEight'
	date_column = 'date'


class Source2:
	leagues = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'F1', 'N1', 'SP1', 'P1']
	seasons = [1617, 1718, 1819, 1920, 2021]
	source_link = 'http://www.football-data.co.uk/mmz4281'
	links = []
	for league in leagues:
		for season in seasons:
			links.append(f'{source_link}/{season}/{league}.csv')

	save_folder = 'Football-data/data/'
	date_column = 'Date'

	columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 
				'shots1', 'shots2', 'shotsot1', 'shotsot2', 
				'fouls1', 'fouls2', 'corners1', 'corners2', 
				'yellow1', 'yellow2', 'red1', 'red2', 
				'MaxH', 'MaxD','MaxA', 'AvgH', 'AvgD', 'AvgA', 
				'Max>2.5', 'Max<2.5', 'Avg>2.5', 'Avg<2.5']

	rename_dict = {'BbMxH': 'MaxH', 'BbMxD': 'MaxD', 
					'BbMxA': 'MaxA', 'BbAvH': 'AvgH', 
					'BbAvD': 'AvgD', 'BbAvA': 'AvgA',
					'HS': 'shots1', 'AS': 'shots2', 
					'HST': 'shotsot1', 'AST': 'shotsot2', 
					'HF': 'fouls1', 'AF': 'fouls2', 
					'HC': 'corners1', 'AC': 'corners2', 
					'HY': 'yellow1', 'AY': 'yellow2', 
					'HR': 'red1', 'AR': 'red2',
					'BbMx>2.5': 'Max>2.5', 
					'BbMx<2.5': 'Max<2.5', 
					'BbAv>2.5': 'Avg>2.5', 
					'BbAv<2.5': 'Avg<2.5'}