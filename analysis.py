import pandas as pd
import numpy as np
from scipy import stats

class FeatureTransformer:

	def __init__(self):
		pass


	def fit(self, X):
		skews_original = (3 * (X.mean() - X.median())) / X.std()
		skews_original = skews_original.rename('orig')

		log = np.log(X + 0.01)
		skews_log = (3 * (log.mean() - log.median())) / log.std()
		skews_log = skews_log.rename('log')

		sqrt = np.sqrt(X + 0.01)
		skew_sqrt = (3 * (sqrt.mean() - sqrt.median())) / sqrt.std()
		skew_sqrt = skew_sqrt.rename('sqrt')

		skews = pd.concat([skews_original, skews_log, skew_sqrt], axis=1)
		self.transformations_ = skews.idxmin(axis=1).to_dict()


		return self


	def transform(self, X):
		X_transformed = pd.DataFrame()
		for column in X.columns:
			if self.transformations_[column] == 'log':
				X_transformed[column] = np.log(X[column] + 0.01)
			elif self.transformations_[column] == 'sqrt':
				X_transformed[column] = np.sqrt(X[column] + 0.01)
			else:
				X_transformed[column] = X[column]

		return X_transformed

	@property
	def transformations(self):
		return self.transformations_