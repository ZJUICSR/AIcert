import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import autokeras as ak
from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import numpy as np
 
tBatchSize = 128

model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.h5',custom_objects=ak.CUSTOM_OBJECTS)
 
'''第四步：训练'''
 
def mnist_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
(x_train, y_train), (X_test, Y_test) = mnist_load_data()

 
 

print("test set")
# 误差评价 ：按batch计算在batch用到的输入数据上模型的误差
scores = model.evaluate(X_test,Y_test, batch_size=tBatchSize, verbose=0)
print("")
print("The test loss is %f" % scores)
 
# 根据模型获取预测结果  为了节约计算内存，也是分组（batch）load到内存中的，
result = model.predict(X_test,batch_size=tBatchSize,verbose=1)
 
# 找到每行最大的序号
 #axis=1表示按行 取最大值   如果axis=0表示按列 取最大值 axis=None表示全部
result_max = np.argmax(result, axis = 1)
 # 这是结果的真实序号
test_max = np.argmax(Y_test, axis = 1)

 
result_bool = np.equal(result_max, test_max) # 预测结果和真实结果一致的为真（按元素比较）
true_num = np.sum(result_bool) #正确结果的数量
print("The accuracy of the model is %f" % (true_num/len(result_bool))) # 验证结果的准确率
