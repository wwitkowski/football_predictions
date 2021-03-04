import pandas as pd
import numpy as np
from scipy import stats

class Distribution:

	def __init__(self):
		pass

	@staticmethod
	def straighten(series):
		skews = {}
		skew_original = (3 * (series.mean() - series.median())) / series.std()
		skews[abs(skew_original)] = series

		log = np.log(series)
		skew_log = (3 * (log.mean() - log.median())) / log.std()
		skews[abs(skew_log)] = log

		sqrt = np.sqrt(series)
		skew_sqrt = (3 * (sqrt.mean() - sqrt.median())) / sqrt.std()
		skews[abs(skew_sqrt)] = sqrt

		boxcox, _ = stats.boxcox(series)
		skew_boxcox = (3 * (boxcox.mean() - np.median(boxcox))) / boxcox.std()
		skews[abs(skew_boxcox)] = boxcox

		#print([abs(key) for key in skews.keys()])
		min_skew = min(skews.keys())

		return skews[min_skew]
