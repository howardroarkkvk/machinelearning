from sklearn.feature_selection import f_classif
import pandas as pd

df = pd.DataFrame({
    'Salary': [20, 24, 80, 85],
    # 'Age':[22,25,40,45],
    'Department': [0, 0,1, 1]  # encoded
})

print(df)
X = df[['Salary']]
y = df['Department']

f_score, p_value = f_classif(X, y)

print("F-score:", f_score)
print("p-value:", p_value)


# variance with in the group/variance between the group
# in this groups are done basedon department.....
#    Salary  Department
# 0      20           0
# 1      24           0
# 2      80           1
# 3      85           1
# here 20,24 is one group and 80,85 is another 
#k - no. of groups -> 2
# N - total no .of observations -> 4

#with in group variance
# mean of group 1 22, mean of group 2 82.5
# variance of group 1 (20-22)^2+(24-22)^2 -> 8
# varaince of group 2 (80-82.5)^2+(85-82.5)^2-> 12.5
# MSW: sum of varaince of group 1 and 2 = 8+12.5=>20.5

# between groups variances
# mean of group 1 8 and group 2 12.5
# overall group mean 20,24,80,85 -> 209/4 -> 52.25 is the mean of overall group
# 2(52.5-22)^2+2(52.5-82.5)^2 => 3660.5 
# MSB= 3660.5/(k=2)-1 , here k is no. of groups in X variable
# MSW=20.5/4-2=> 10..25

#F=MSB/MSW => 3660.5/10.25 -> 357.1





