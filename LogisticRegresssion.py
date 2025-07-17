import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import log_loss



file_dir=r'D:\ML_data\Logistic_Regression\stars'
file_name='train.csv'
star_df=pd.read_csv(os.path.join(file_dir,file_name))


# for file in Path(file_dir).iterdir():
#     print(file.name)

print('Top 5 records from a dataframe using head():',star_df.head())
print('Info of all the columns using info(): ',star_df.info())
print('describe of all the columns using describe(): \n',star_df.describe())
print('List all columns using columns:',star_df.columns)


print(star_df[['Spectral Class']].value_counts())

# print(star_df.head().value_counts())

star_df.replace({'Spectral Class':{'M':0,'A':1,'B':1,'F':1,'O':1,'K':1,'G':1}},inplace=True)
print(star_df[['Spectral Class']].value_counts())


print(star_df['Star type'].value_counts())

print(star_df['Star color'].value_counts())

star_df.replace({'Star color':{'Red':0,'Yellow':1,'White':2,'Blue':3}},inplace=True)

print(star_df['Star color'].value_counts())

X=star_df[['Temperature (K)','Luminosity (L/Lo)', 'Radius (R/Ro)','Absolute magnitude (Mv)']]
y=star_df['Spectral Class']


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)

print('X  train \n',X_train.head())
print('X  test \n',X_test.head())
print('y  train \n',y_train.head())
print('y  test \n',y_test.head())



model=LogisticRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
y_pred_2d=y_pred.reshape(-1,1)

# print('y  predicted \n',y_pred_2d)

y_test_np=y_test.to_numpy().reshape(-1,1)
overall_loss=log_loss(y_pred_2d,y_test_np)
print('Loss for the given dataset is : ',overall_loss)
# print('y  test  np\n',y_test_np)
pred_actual_test_np_arr=np.concatenate((y_pred_2d,y_test_np),axis=1)
# print(pred_actual_test_np_arr)

pred_actual_test_df=pd.DataFrame(pred_actual_test_np_arr,columns=['predicted','actual'])
print(pred_actual_test_df)

cm=confusion_matrix(y_test,y_pred)
print(cm)

print(classification_report(y_test,y_pred))