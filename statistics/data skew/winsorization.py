
from scipy.stats.mstats import winsorize
import numpy as np

data = np.array([10, 12, 11, 13, 9, 100])

# cap lowest 5% and highest 5%
# winsorized = winsorize(data, limits=[0.02, 0.02])

lower=np.percentile(data,5)
print(lower)
upper=np.percentile(data,95,method='lower')
print(upper)
winsorized = np.clip(data,lower,upper)

print(winsorized)