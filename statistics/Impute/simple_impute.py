# model assumes order....so it is bad for nominal data...
# use this for tree models as they dont care about order, it is useful for ordinal cat columns where simple, medium and complex like order is present

import pandas as pd
from  sklearn.preprocessing import LabelEncoder,OneHotEncoder

df = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue']})

# label encoding
# le=LabelEncoder()
# df['color_encoded']=le.fit_transform(df['color'])

# one hot encoding
ohe=OneHotEncoder(sparse_output=False)
df1=ohe.fit_transform(df[['color']])

print(ohe.get_feature_names_out())
one_hot_df=pd.DataFrame(df1,columns=ohe.get_feature_names_out())

print(df)
print(df1)
print(one_hot_df)
print(pd.concat([df,one_hot_df],axis=1))

# ordinal encoding - order matters hence we should pass the order mapping as input

df2 = pd.DataFrame({'size': ['small', 'medium', 'large']})

mapping = {'small': 1, 'medium': 2, 'large': 3}

df2['size_encoded']=df2['size'].map(mapping)
print(df2)