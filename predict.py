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

ds2_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST', 
		'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'MaxH', 'MaxD','MaxA', 
		'AvgH', 'AvgD', 'AvgA']
ds2_rename_dict = {'BbMxH': 'MaxH', 'BbMxD': 'MaxD', 'BbMxA': 'MaxA', 'BbAvH': 'AvgH', 'BbAvD': 'AvgD', 'BbAvA': 'AvgA'}

ds2 = Dataset(source=source2, save_folder='Football-data/data/', 
			columns=ds2_columns, date_column='Date',
			rename_dict=ds2_rename_dict, sub_folder=True)

ds2.data.to_csv('ds2.csv')
print(ds1.last_update_date)
print(ds2.last_update_date)
