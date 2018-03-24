from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np
data=np.random.random((1000,100))
print(data)
labels=np.random.randint(2,size=(1000,1))
#模型搭建阶段
model=Sequential()#代表类的初始化
model.add(Dense(32,activation='relu',input_dim=100))#代表全连接层
# ，此时有32个全连接层，最后接relu，输入的是100维度
# Dense(32) is a fully-connected layer with 32 hidden units.
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
model.fit(data,labels,epochs=10,batch_size=32)
