import matplotlib.pyplot as plt

print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import*
from sklearn.model_selection import train_test_split

####Step 1
path = 'myData'
data = importDataInfo(path)

###Step 2
data = balanceData(data,display=False)

###Step 3
imagesPath, steerings = loadData(path,data)
#print(imagesPath[0], steering[0])

###Step 4
xTrain,xVal,yTrain,yVal = train_test_split(imagesPath,steerings,test_size=0.2,random_state=5)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

###Step 5

###Step 6

###Step 7

###Step 8
model = createModel()
model.summary()

###Step 9
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=300,epochs=10,
          validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

###Step 10
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

