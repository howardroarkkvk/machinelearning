import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import LinearSVC,SVC
from tabulate import tabulate

cancer=load_breast_cancer()
print(cancer.data,cancer.data.shape)
print(cancer.target,cancer.target.shape)
col_names=list(cancer.feature_names)
if 'target' not in col_names:
    print('target is not present in col_names')
print(col_names,type(col_names))
col_names.append('target')
df=pd.DataFrame(np.c_[cancer.data,cancer.target],columns=col_names)
print(df.head(5))
print(cancer.target_names)
print(df.describe())
print(df.info())


y=df.target
X=df.drop('target',axis=1)
print(f"'X' shape is : {X.shape}")
print(f"'y' shape is : {y.shape}")

pipeline=Pipeline([('min_max_scaler',MinMaxScaler()),('std_scaler',StandardScaler())])
print(pipeline)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

print('finding if nulls are present in training data ',X_train.isnull().any())
print('finding if nulls are present in target data ',y_train.isnull().any())
print(y_train.to_numpy())


model=LinearSVC(loss='hinge',dual=True)
model.fit(X_train,y_train)

def print_score(model,X_train,y_train,X_test,y_test,train=True):
    print(" This is in print function ")
    print(" *********************************** ")
    if train:
        # pred=model.predict(X_train)
        acc_score=accuracy_score(y_train,pred)
        print(f"This is accuracy score {acc_score*100:.2f}")
        print(" *********************************** ")

        conf_matrix=confusion_matrix(y_train,pred)
        conf_matrix_pd=pd.DataFrame(conf_matrix,columns=['class 0 predicated','class 0 predicated'],index=['class 0 actual','class 1 actual'])
        print(conf_matrix_pd)
        print(" *********************************** ")

        classification_rpt=classification_report(y_train,pred,output_dict=True,target_names=['class0','class1'])
        # print(classification_rpt,type(classification_rpt))
        classification_rpt_df=pd.DataFrame(classification_rpt)
        classification_rpt_df_transpose=classification_rpt_df.T
        print(tabulate(classification_rpt_df_transpose,headers='keys',tablefmt='fancy_grid',stralign='center',numalign='center'))#fancy_grid,grid,md,html,LaTeX
        print(" *********************************** ")


pred=model.predict(X_train)
print(pred,pred.shape)
print_score(model,X_train,y_train,X_test,y_test,True)
# pred_df=pd.DataFrame(pred,columns=['pred'])
# print(pred_df,pred_df.shape)
# y_train_df=pd.DataFrame(y_train.to_numpy())
# print(y_train_df,y_train_df.shape)
# pred_vs_y_train_df=pd.concat([y_train_df,pred_df],axis=1)
# pred_vs_y_train_df.columns=['y_train','pred']

# pred_vs_y_train_df['y_train_eq_pred']=(pred_vs_y_train_df['y_train']==pred_vs_y_train_df['pred'])
# print(pred_vs_y_train_df)
# print(pred_vs_y_train_df.value_counts('y_train_eq_pred'))

# acc_score=accuracy_score(y_train,pred)
# print(f"{acc_score*100:.2f},{type(acc_score)}")

# conf_matrix=confusion_matrix(y_train,pred)
# conf_matrix_pd=pd.DataFrame(conf_matrix,columns=['class 0 predicated','class 0 predicated'],index=['class 0 actual','class 1 actual'])
# print(conf_matrix_pd)

# classification_rpt=classification_report(y_train,pred,output_dict=True,target_names=['class0','class1'])
# # print(classification_rpt,type(classification_rpt))
# classification_rpt_df=pd.DataFrame(classification_rpt)
# classification_rpt_df_transpose=classification_rpt_df.T
# print(tabulate(classification_rpt_df_transpose,headers='keys',tablefmt='fancy_grid',stralign='center',numalign='center'))#fancy_grid,grid,md,html,LaTeX




model_rbf=SVC(kernel='rbf',gamma=0.5,C=0.1)
model_rbf.fit(X_train,y_train)
print('This is rbf model call..')
print_score(model_rbf,X_train,y_train,X_test,y_test,True)



model_poly=SVC(kernel='poly',degree=2,gamma='auto',coef0=1,C=5)
model_rbf.fit(X_train,y_train)
print('This is poly model call..')
print_score(model_rbf,X_train,y_train,X_test,y_test,True)


# model1=SVC(kernel='linear',probability=True)
# X=cancer.data
# x=X[:,:]
# x_min=x[:,0].min()-1
# x_max=x[:,0].min()+1
# y=cancer.target
# y_min=y.min()-1
# y_max=y.max()+1

# df1=pd.DataFrame(x[:,0:2],columns=['feature1','feature2'])
# print(df1.head(),df1.shape)
# df1['target']=y


# model1.fit(x[:,0:2],y)
# xx,yy=np.meshgrid(np.linspace(x_min,x_max,300),                
#             np.linspace(y_min,y_max,300))

# print(xx.shape,yy.shape)
# print(xx.ravel(),xx.ravel().shape) # flattenning of the x to 1d

# grid=np.c_[xx.ravel(),yy.ravel()]
# print(grid.shape)
# probs=model1.predict_proba(grid)[:,1].reshape(xx.shape)
# print(probs.shape)
# print(probs[0:5,:])
# plt.figure(figsize=(10,6))
# contour=plt.contourf(xx,yy,probs,25,cmap='coolwarm',alpha=0.8)
# sns.scatterplot(data=df1,x='feature1',y='feature2',hue='target',palette='coolwarm',edgecolor='k')
# plt.title('Svm decisoin boundary linear kernel')
# plt.xlabel('feature1')
# plt.ylabel('feature2')
# plt.colorbar(contour)
# plt.show()






# model1=SVC(kernel='poly',degree=2,gamma='auto',coef0=1,C=5)

# sns.pairplot(df,hue='target',vars=['mean radius','mean texture','worst area'],diag_kind='hist',)
# plt.show()

# sns.countplot(x=df['target'],label='count')
# plt.show()

# plt.figure(figsize=(10,8))
# sns.scatterplot(x='mean area',y='mean smoothness',hue='target',data=df)
# plt.show()

# print(df.corr())
# plt.figure(figsize=(20,10))
# sns.heatmap(df.corr(),annot=True)
# plt.show()

#SPLASH JB
# sns.scatterplot(data=df,x='mean radius',y='worst area',hue='target')
# plt.show()


# sns.lmplot(df,x='mean radius',y='target')
# plt.show()

# sns.regplot(data=df,x='mean radius',y='target')
# plt.show()

# sns.jointplot(data=df,x='mean radius',y='target')
# plt.show()

# sns.swarmplot(data=df,x='mean radius',y='target')
# plt.show()

# sns.kdeplot(data=df,x='mean radius',y='target')
# plt.show()

# sns.countplot(x=df['target'],label='count')
# plt.show()
