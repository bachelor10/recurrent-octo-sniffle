import numpy as np
import keras 
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')

CLASS_INDICES = {'3': 7, 'y': 36, 'lt': 26,'\lt': 26, 'gamma': 22, '\\gamma': 22, 'beta': 20, '\\beta': 20, ')': 1, '0': 4, '1': 5, 'sqrt': 33, '\sqrt': 33, 'lambda': 25, '\\lambda': 25, '7': 11, 'z': 37, '6': 10, 'Delta': 15,'\\Delta': 15, '-': 3, 'neq': 28,'\\neq': 28, '=': 14, '8': 12, 'G': 16, 'sigma': 32,'\\sigma': 32, 'f': 21, 'rightarrow': 31,'\\rightarrow': 31, 'phi': 29,'\phi': 29, 'infty': 24,'\infty': 24, 'x': 35, '[': 17, '9': 13, 'gt': 23, '\gt': 23, 'theta': 34,'\\theta': 34, 'pi': 30, '\pi': 30, '4': 8, '5': 9, '2': 6, 'mu': 27, '\mu': 27, '(': 0, ']': 18, 'alpha': 19, '\\alpha': 19, '+': 2}

def read_validation_files():
    filename = "../../data/"

    train_img = np.load(filename + "trainX_img.npy")
    train_trace = np.load(filename + "trainX_trace.npy")
    train_y = np.load(filename + 'trainY.npy')
    
    return [train_trace, train_img], train_y

def vizualize_epoch_logs(acc, loss, val_acc, val_loss, real_acc, real_loss, title):
    plt.plot(acc)
    plt.plot(val_acc)
    plt.plot(real_acc)
    plt.ylim([0.85, 1])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'real'], loc='upper left')
    plt.savefig('./images/' + title)

def truth_from_index(index):
    for key, value in CLASS_INDICES.items():
        if value == index: #41, 
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


def count_misclassification(predictions, trainY):
    misclassifications = dict()

    for x, y in zip(predictions, trainY):
        if(truth_from_index(np.argmax(x)) != truth_from_index(np.argmax(y))):
            truth = truth_from_index(np.argmax(y))
            if truth not in misclassifications:
                misclassifications[truth] = 0
            misclassifications[truth] += 1
    return misclassifications
    print("Misclassifications", misclassifications)


model = keras.models.load_model("../../data/combined_model.h5")

trainX, trainY = read_validation_files()

print(trainY.shape)
result = model.predict(trainX)

misclassifications = count_misclassification(result, trainY)

misclassifications = sorted(misclassifications.items(), key=lambda kv: kv[1], reverse=True)
print(misclassifications)


plt.bar([miss[0] for miss in misclassifications][:10], [miss[1] for miss in misclassifications][:10])
plt.title("Misclassification")
plt.ylabel('Number of misclassifications')
plt.xlabel('Symbol')
plt.rc('font', size=12)          # controls default text sizes

plt.show()
"""
vizualize_epoch_logs(acc, loss, val_acc, val_loss, real_acc, real_loss, "Combined CNN and RNN (44000)")
plt.figure()
vizualize_epoch_logs(acc1, loss1, val_acc1, val_loss1, real_acc1, real_loss1, "CNN (44000)")
plt.figure()
vizualize_epoch_logs(acc2, loss2, val_acc2, val_loss2, real_acc2, real_loss2, "RNN (44000)")
plt.show()"""
