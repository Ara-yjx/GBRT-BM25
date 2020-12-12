from feature_reader import features, values, comments
from catboost import Pool, CatBoostRegressor
import time



features = features[:, (0,1,2,3,4,5,6,7,8,24,25,26,27,28,29,30,31,32,33,34,35)]
print(features.shape)




data_size = features.shape[0]
split_size = int(data_size*0.8)

train_data = features[:split_size]
train_label = values[:split_size]
test_data = features[split_size:]
test_label = values[split_size:]

train_pool = Pool(train_data, train_label)
test_pool = Pool(test_data)

model = CatBoostRegressor(iterations=5, 
                          depth=5, 
                          learning_rate=0.2, 
                          loss_function='RMSE')

model.fit(train_pool)



time_start=time.time()
preds = model.predict(test_pool)
time_end=time.time()
print(preds)
print(preds.shape)
print('time: ', time_end-time_start,'s')

mse = ((preds - test_label)**2).mean()
print('mse: ', mse)


test_qid = comments[split_size:]
currentQid = ''
for i in range(data_size-split_size):
    if test_qid[i] != currentQid:
        currentQid = test_qid[i]
        print(currentQid)
    print(str(test_label[i]) + ',' + str(preds[i]))
    
        

