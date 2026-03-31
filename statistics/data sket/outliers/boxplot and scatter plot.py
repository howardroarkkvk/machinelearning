import seaborn as sns
import matplotlib.pyplot as plt

prices = [0.05,10, 20, 30, 40, 50, 60, 65, 70, 200]


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.boxplot(y=prices)
plt.title('Seaborn box plot - outliers')


plt.subplot(1,2,2)
area = [1000, 1200, 1100, 1300, 900, 1050, 1150, 1250, 1400]
prices = [10, 12, 11, 13, 9, 10, 11, 12, 200]

sns.scatterplot(x=area, y=prices)
plt.title("Seaborn Scatter Plot - Outlier Detection")
plt.show()