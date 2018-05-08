import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

CLASS_INDICES = {'3': 7, 'y': 36, 'lt': 26,'\lt': 26, 'gamma': 22, '\\gamma': 22, 'beta': 20, '\\beta': 20, ')': 1, '0': 4, '1': 5, 'sqrt': 33, '\sqrt': 33, 'lambda': 25, '\\lambda': 25, '7': 11, 'z': 37, '6': 10, 'Delta': 15,'\\Delta': 15, '-': 3, 'neq': 28,'\\neq': 28, '=': 14, '8': 12, 'G': 16, 'sigma': 32,'\\sigma': 32, 'f': 21, 'rightarrow': 31,'\\rightarrow': 31, 'phi': 29,'\phi': 29, 'infty': 24,'\infty': 24, 'x': 35, '[': 17, '9': 13, 'gt': 23, '\gt': 23, 'theta': 34,'\\theta': 34, 'pi': 30, '\pi': 30, '4': 8, '5': 9, '2': 6, 'mu': 27, '\mu': 27, '(': 0, ']': 18, 'alpha': 19, '\\alpha': 19, '+': 2}

def read_data_files():
    trainX = [
        np.load('../online_recog/data/trainX_trace.npy'),
        np.load('../online_recog/data/trainX_img.npy')
    ]
    trainY = np.load('../online_recog/data/trainY.npy')
    original_traces = np.load('../online_recog/data/original_traces.npy')

    realX = [
        np.load('../online_recog/data/real_test_data/trainX_trace.npy'),
        np.load('../online_recog/data/real_test_data/trainX_img.npy')
    ]
    realY = np.load('../online_recog/data/real_test_data/trainY.npy')

    return trainX, trainY, realX, realY, original_traces


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


def vizualize_data(trainX, trainY, count=100):
    for i in range(count):
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.array(trainX[1][i]).reshape(26, 26))
        ax2.plot(trainX[0][i][:, 0], trainX[0][i][:, 1], '-o')
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([1, -1])
        if i == 3:
            plt.savefig('./images/beta_processed.png')
        plt.show()

def filter_by_truth(trainY, truth):

    indices = np.zeros(len(trainY), dtype=bool)

    for i in range(len(trainY)):
        if truth_from_index(np.argmax(trainY[i])) == truth:
            indices[i] = True

    return indices

def plot_original_traces(traces):
    for trace in traces:
        trace = np.array(trace)
        plt.plot(trace[:, 0], trace[:, 1], '-o')
        
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])  
  
def plot_image(image):
    plt.imshow(np.array(image).reshape(26, 26))


def remove_padding(trace_group):
    original_trace_group = []

    current_trace = []
    for i, coords in enumerate(trace_group):
        if(coords[0] != 0 or coords[1] != 0 or coords[2] != 0):
            current_trace.append([coords[0], coords[1]])
        if(coords[2] == 1 or i == len(trace_group) - 1):
            original_trace_group.append(current_trace)
            current_trace = []
    
    return original_trace_group
            
def generate_sqrt_example(trainX, trainY, original_traces):
    sqrt_indices = filter_by_truth(trainY, '\\sqrt')
    
    for i, padded_trace_group in enumerate(trainX[0]):
        if not sqrt_indices[i]: continue
        if(len(np.array(original_traces[i])[0][:, 0]) != 242): continue
        unpadded_trace_group = remove_padding(padded_trace_group)


        plot_original_traces(np.array(unpadded_trace_group))
        original_length = len(np.array(original_traces[i])[0][:, 0])
        preprocessed_length = len(np.array(unpadded_trace_group)[0][:, 0])
        plt.title("After preprocessing (Num coords = " + str(preprocessed_length) + ")")

        plt.figure()
        plt.title("Before preprocessing (Num coords = " + str(original_length) + ")")
        plot_original_traces(np.array(original_traces[i]))

        plt.figure()
    
        plt.title("Image generated from traces")
        plot_image(trainX[1][i])

        plt.show()

def compareTrainToReal(train_img, real_img):
    for train, real in zip(train_img, real_img):
        plot_image(train)
        plt.figure()
        plot_image(real)
        plt.show()

def inspect_images(imgs, truths):
    for i, image in enumerate(imgs):
        plot_image(image)
        plt.title("Image type: " + str(truth_from_index(np.argmax(truths[i]))) + ". Image number: " + str(i))
        plt.show() #Slett 41, 174, 

trainX, trainY, realX, realY, original_traces = read_data_files()

inspect_images(realX[1], realY)
#train_indices = filter_by_truth(trainY, '\\beta')
#real_indices = filter_by_truth(realY, '\\beta')

#compareTrainToReal(trainX[1][train_indices], realX[1][real_indices])

#generate_sqrt_example(trainX, trainY, original_traces)
#realX[0] = realX[0].reshape(len(realX[0]), 40, 3)

#vizualize_data(realX, realY)

#filtered_x, filtered_y, original_traces = filter_by_truth(trainX, trainY, original_traces, "\\beta")

#vizualize_data(filtered_x, filtered_y)
#plot_original_traces(original_traces[0])



