from sklearn.preprocessing import MinMaxScaler,RobustScaler
data = [[10],[20],[30],[40],[50],[60]]

scaler=MinMaxScaler()
scaled=scaler.fit_transform(data)
print(scaled)

# x - xmin / xmax-xmin
robust=RobustScaler()
robusted=robust.fit_transform(data)
print(robusted)