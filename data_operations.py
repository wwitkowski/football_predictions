import pandas as pd
import numpy as np
import math
import os
import re
from datetime import datetime


class DateHandler:

	def __init__(self):
		pass

	def to_date(date_string):
		splitter = re.search("[./-]", date_string)[0]
		groups = re.split("([./-])", date_string)
		date_len = len(date_string)
		if len(groups[0]) == 2 and date_len == 8:
			return datetime.strptime(date_string, f'%d{splitter}%m{splitter}%y')
		elif len(groups[0]) == 4 and date_len == 10:
			return datetime.strptime(date_string, f'%Y{splitter}%m{splitter}%d')
		else:
			return datetime.strptime(date_string, f'%d{splitter}%m{splitter}%Y')



class DatasetLoader:

	def __init__(self, source, save_folder):
		self.source = source
		self.save_folder = save_folder

	
	def load(self, link, sub_folder=False):
		file_name = link.split('/')[-1]
		if sub_folder:
			sub_folder = link.split('/')[-2]
			path = f'{self.save_folder}/{sub_folder}/{file_name}'
		else:
			path = f'{self.save_folder}/{file_name}'
		try:
			data = pd.read_csv(path)
		except FileNotFoundError:
			try:
				data = pd.read_csv(link)
				data.to_csv(path, index=False)
			except Exception as e:
				print(f'Could not download the file {path}. {e}')

		return data


class Dataset(DatasetLoader):

	def __init__(self, source, save_folder, date_column, columns=None, rename_dict=None, sub_folder=False):
		super().__init__(source, save_folder)
		self.data = pd.DataFrame()
		self.columns = columns
		self.rename_dict = rename_dict
		self.sub_folder = sub_folder

		self.concat_dataframes()
		self.data.rename(columns={date_column: 'date'}, inplace=True)
		self.data['date'] = self.data['date'].apply(lambda x: DateHandler.to_date(x).strftime('%Y-%m-%d'))

	
	def concat_dataframes(self):
		for link in self.source:
			temp_df = self.load(link, self.sub_folder)
			if self.columns is not None:
				try:
					temp_df = temp_df[self.columns]
				except KeyError:
					temp_df.rename(columns=self.rename_dict, inplace=True)
					try:
						temp_df = temp_df[self.columns]
					except KeyError as e:
						print(f'{link}: {e} - data skipped.')
						continue

			self.data = pd.concat([self.data, temp_df], sort=False, ignore_index=True)

	
	def one_day(self, date):
		"""
		Function returning matches for a specific matchday.
		"""
		return self.data[self.data.date == date]


	@property
	def last_update_date(self):
		"""
		Returns last update date. The date is found where the last not null value is.
		"""
		last_valid_idxs = self.data.apply(pd.Series.last_valid_index)
		last_update_idx = min(last_valid_idxs.values)

		return self.data.date.iloc[last_update_idx]

