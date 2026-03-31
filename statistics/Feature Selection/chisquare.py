import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Male', 'Female','Male', 'Female', 'Male', 'Female','Male', 'Female', 'Male', 'Female','Male', 'Female', 'Male', 'Female'],
    'Target': ['Yes', 'No', 'Yes', 'No','Yes', 'Yes','Yes', 'No','No', 'Yes','No', 'Yes','No', 'Yes','No', 'Yes']
})



print(df)


le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Target']=le.fit_transform(df['Target'])

print(df)
t=pd.crosstab(df['Gender'],df['Target'])
print(t,type(t),t.info(),t.columns)
X=df[['Gender']]
print(X)

y=df['Target']

chi_score,p_values=chi2(X,y)
print(chi_score,p_values)

# observered  Target     expected Target
#        0    1             0        1
#0- F    3    5  -> 8     (8*11/16)5.5   (8*5/16)2.5
#1- M    8    0  -> 8     (8*11/16)5.5  (8*5/16) 2.5
#------------------
#        11   5   => 16

# (3-5.5)^2/5.5+ (5-2.5)^2/2.5 + (8-5.5)^2/5.5+(0-2.5)^2/2.5
# 6.25/5.5+6.25/2.5+6.25/5.5+6.25/2.5
# 7.27

#degrees of feedom -> (2-1)*(2-1) -> 1

# only 5% it occured randomly....