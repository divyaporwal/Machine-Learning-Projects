import tensorflow as tf
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D
import matplotlib.pyplot as plt
import math
import pickle

def build_model(n_kernels=8, kernel_size=3, stride=2, n_dense=32):
    model = Sequential()
    model.add(Conv2D(n_kernels, (kernel_size, kernel_size), activation='relu', input_shape=(16, 16, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(stride, stride)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=n_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    #Open usps dataset
    data = pickle.load(open('usps.pickle', 'rb'))
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

    # n_kernel values
    kernelArray = [1, 2, 4, 8, 16]
    
    # kernel size values
    kernelSize = [1, 2, 3, 4, 5]
    
    # stride values
    strideArray = [1, 2, 3, 4]
    
    # n_dense values
    denseArray = [16, 32, 64, 128]

    testingErrorArray = []
    trainingErrorArray = []
    modelParams = []
    for k in kernelArray:
        model = build_model(k, 3, 2, 32)
        history = model.fit(data['x']['trn'], data['y']['trn'], epochs=2, batch_size=16, verbose=2, validation_data=(data['x']['val'], data['y']['val']), callbacks=[annealer])
        trn_score = model.evaluate(data['x']['trn'], data['y']['trn'], batch_size=16)
        tst_score = model.evaluate(data['x']['tst'], data['y']['tst'], batch_size=16)
        trainingErrorArray.append(trn_score[0])
        testingErrorArray.append(tst_score[0])
        modelParams.append(model.count_params())
    
    plt.plot(kernelArray, trainingErrorArray, label="Training Error", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.plot(kernelArray, testingErrorArray, label="Test Error", marker ='o', markerfacecolor ='b', markeredgecolor ='b', linestyle =':')
    plt.xlabel('n-kernel x-axis')
    plt.ylabel('Error y-axis')
    plt.legend(loc='upper right')
    plt.title('n-kernel vs Error')
    plt.show()
    
    plt.plot(kernelArray, modelParams, label="Model Params", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.xlabel('n-kernel x-axis')
    plt.ylabel('Model Params y-axis')
    plt.legend(loc='upper right')
    plt.title('n-kernel vs Model Params')
    plt.show()
    
    testingErrorArray = []
    trainingErrorArray = []
    modelParams = []
    for k in kernelSize:
        model = build_model(8, k, 2, 32)
        history = model.fit(data['x']['trn'], data['y']['trn'], epochs=2, batch_size=16, verbose=2, validation_data=(data['x']['val'], data['y']['val']), callbacks=[annealer])
        trn_score = model.evaluate(data['x']['trn'], data['y']['trn'], batch_size=16)
        tst_score = model.evaluate(data['x']['tst'], data['y']['tst'], batch_size=16)
        trainingErrorArray.append(trn_score[0])
        testingErrorArray.append(tst_score[0])
        modelParams.append(model.count_params())
    
    plt.plot(kernelSize, trainingErrorArray, label="Training Error", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.plot(kernelSize, testingErrorArray, label="Test Error", marker ='o', markerfacecolor ='b', markeredgecolor ='b', linestyle =':')
    plt.xlabel('Kernel Size x-axis')
    plt.ylabel('Error y-axis')
    plt.legend(loc='upper right')
    plt.title('Kernel Size vs Error')
    plt.show()
    
    plt.plot(kernelSize, modelParams, label="Model Params", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.xlabel('Kernel Size x-axis')
    plt.ylabel('Model Params y-axis')
    plt.legend(loc='upper right')
    plt.title('Kernel Size vs Model Params')
    plt.show()
    
    testingErrorArray = []
    trainingErrorArray = []
    modelParams = []
    for k in strideArray:
        model = build_model(8, 3, k, 32)
        history = model.fit(data['x']['trn'], data['y']['trn'], epochs=2, batch_size=16, verbose=2, validation_data=(data['x']['val'], data['y']['val']), callbacks=[annealer])
        trn_score = model.evaluate(data['x']['trn'], data['y']['trn'], batch_size=16)
        tst_score = model.evaluate(data['x']['tst'], data['y']['tst'], batch_size=16)

        trainingErrorArray.append(trn_score[0])
        testingErrorArray.append(tst_score[0])
        modelParams.append(model.count_params())
    
    plt.plot(strideArray, trainingErrorArray, label="Training Error", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.plot(strideArray, testingErrorArray, label="Test Error", marker ='o', markerfacecolor ='b', markeredgecolor ='b', linestyle =':')
    plt.xlabel('Stride x-axis')
    plt.ylabel('Error y-axis')
    plt.legend(loc='upper right')
    plt.title('Stride vs Error')
    plt.show()
    
    plt.plot(strideArray, modelParams, label="Model Params", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.xlabel('Stride x-axis')
    plt.ylabel('Model Params y-axis')
    plt.legend(loc='upper right')
    plt.title('Stride vs Model Params')
    plt.show()
    
    testingErrorArray = []
    trainingErrorArray = []
    modelParams = []
    for k in denseArray:
        model = build_model(8, 3, 2, k)
        history = model.fit(data['x']['trn'], data['y']['trn'], epochs=2, batch_size=16, verbose=2, validation_data=(data['x']['val'], data['y']['val']), callbacks=[annealer])
        trn_score = model.evaluate(data['x']['trn'], data['y']['trn'], batch_size=16)
        tst_score = model.evaluate(data['x']['tst'], data['y']['tst'], batch_size=16)

        trainingErrorArray.append(trn_score[0])
        testingErrorArray.append(tst_score[0])
        modelParams.append(model.count_params())
    
    plt.plot(denseArray, trainingErrorArray, label="Training Error", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.plot(denseArray, testingErrorArray, label="Test Error", marker ='o', markerfacecolor ='b', markeredgecolor ='b', linestyle =':')
    plt.xlabel('n_dense x-axis')
    plt.ylabel('Error y-axis')
    plt.legend(loc='upper right')
    plt.title('n_dense vs Error')
    plt.show()
    
    plt.plot(denseArray, modelParams, label="Model Params", marker ='o', markerfacecolor ='r', markeredgecolor ='k', linestyle ='-')
    plt.xlabel('n_dense x-axis')
    plt.ylabel('Model Params y-axis')
    plt.legend(loc='upper right')
    plt.title('n_dense vs Model Params')
    plt.show()
