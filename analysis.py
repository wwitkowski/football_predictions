import pandas as pd
import numpy as np
from scipy import stats

class FeatureTransformer:

	def __init__(self):
		pass


	def fit(self, X):

		self.negative_columns = X.columns[(X < 0).any()]
		self.positive_columns = X.columns[(X >= 0).all()]

		skews_original = (3 * (X.mean() - X.median())) / X.std()
		skews_original = skews_original.rename('orig')

		log_pos = np.log(X[self.positive_columns] + 0.01)
		log_neg = np.log(X[self.negative_columns] + (-1 * X[self.negative_columns].min() + 0.01))
		log = pd.concat([log_pos, log_neg], axis=1)
		skews_log = (3 * (log.mean() - log.median())) / log.std()
		skews_log = skews_log.rename('log')

		sqrt_pos = np.sqrt(X[self.positive_columns] + 0.01)
		sqrt_neg = np.sqrt(X[self.negative_columns] + (-1 * X[self.negative_columns].min() + 0.01))
		sqrt = pd.concat([sqrt_pos, sqrt_neg], axis=1)
		skew_sqrt = (3 * (sqrt.mean() - sqrt.median())) / sqrt.std()
		skew_sqrt = skew_sqrt.rename('sqrt')

		skews = pd.concat([skews_original, skews_log, skew_sqrt], sort=False, axis=1)
		self.transformations_ = skews.idxmin(axis=1).to_dict()


		return self


	def transform(self, X):
		X_transformed = pd.DataFrame()
		for column in X[self.positive_columns].columns:
			if self.transformations_[column] == 'log':
				X_transformed[column] = np.log(X[column] + 0.01)
			elif self.transformations_[column] == 'sqrt':
				X_transformed[column] = np.sqrt(X[column] + 0.01)
			else:
				X_transformed[column] = X[column]
		for column in X[self.negative_columns].columns:
			if self.transformations_[column] == 'log':
				X_transformed[column] = np.log(X[column] + (-1 * X[column].min() + 0.01))
			elif self.transformations_[column] == 'sqrt':
				X_transformed[column] = np.sqrt(X[column] + (-1 * X[column].min() + 0.01))
			else:
				X_transformed[column] = X[column]


		return X_transformed

	@property
	def transformations(self):
		return self.transformations_