import numpy as np
import matplotlib.pyplot as plt

CLASS_INDICES = {'3': 7, 'y': 36, 'lt': 26,'\lt': 26, 'gamma': 22, '\\gamma': 22, 'beta': 20, '\\beta': 20, ')': 1, '0': 4, '1': 5, 'sqrt': 33, '\sqrt': 33, 'lambda': 25, '\\lambda': 25, '7': 11, 'z': 37, '6': 10, 'Delta': 15,'\\Delta': 15, '-': 3, 'neq': 28,'\\neq': 28, '=': 14, '8': 12, 'G': 16, 'sigma': 32,'\\sigma': 32, 'f': 21, 'rightarrow': 31,'\\rightarrow': 31, 'phi': 29,'\phi': 29, 'infty': 24,'\infty': 24, 'x': 35, '[': 17, '9': 13, 'gt': 23, '\gt': 23, 'theta': 34,'\\theta': 34, 'pi': 30, '\pi': 30, '4': 8, '5': 9, '2': 6, 'mu': 27, '\mu': 27, '(': 0, ']': 18, 'alpha': 19, '\\alpha': 19, '+': 2}

def read_data_files():
    trainX = [
        np.load('../online_recog/data/trainX_trace.npy'),
        np.load('../online_recog/data/trainX_img.npy')
    ]
    trainY = np.load('../online_recog/data/trainY.npy')

    realX = [
        np.load('../online_recog/data/real_test_data/trainX_trace.npy'),
        np.load('../online_recog/data/real_test_data/trainX_img.npy')
    ]
    realY = np.load('../online_recog/data/real_test_data/trainY.npy')

    return trainX, trainY, realX, realY


def truth_from_index(index):
    for key, value in CLASS_INDICES.items():
        if value == index:
            return key

def find_truth_index(y, truth, from_index=0):
    index = -1

    for i, one_hot in enumerate(y):
        if i <= from_index: continue 
        thisTruth = truth_from_index(np.argmax(one_hot))

        if thisTruth == truth:
            index = i
            break;
    return index

def vizualize_data(trainX, trainY, count=100):
    for i in range(count):
        print(trainX[1])
        thisTruth = truth_from_index(np.argmax(trainY[i]))
        print("Vizualizing", thisTruth, 'at index', i)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.array(trainX[1][i]).reshape(26, 26))
        ax2.plot(trainX[0][i][:, 0], trainX[0][i][:, 1], '-o')
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([1, -1])

        plt.show()

def filter_by_truth(trainX, trainY, truth):
    trainX_img = []
    trainX_trace = []

    trainY_res = []
    for i in range(len(trainY)):
        if truth_from_index(np.argmax(trainY[i])) == truth:
            trainX_img.append(trainX[1][i])
            trainX_trace.append(trainX[0][i])

            trainY_res.append(trainY[i])

    return [np.array(trainX_trace), np.array(trainX_img)], np.array(trainY_res)

trainX, trainY, realX, realY = read_data_files()

realX[0] = realX[0].reshape(len(realX[0]), 40, 3)

vizualize_data(realX, realY)

print("TrainY", len(trainY))
filtered_x, filtered_y = filter_by_truth(trainX, trainY, "8")

vizualize_data(filtered_x, filtered_y)



