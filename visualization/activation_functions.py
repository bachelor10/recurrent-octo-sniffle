import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


#https://gist.github.com/yusugomori/cf7bce19b8e16d57488a

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T 

def ReLU(x):
    return x * (x > 0)



def draw_activation_functions_dotted(start, stop):

    x = np.arange(start, stop, 0.01)

    relu_plot, = plt.plot(x, ReLU(x), ':',label="ReLU")
    sigmoid_plot, = plt.plot(x, sigmoid(x), '-', label="sigmoid")
    #softmax_plot, = plt.plot(x, softmax(x), label="softmax")
    tanh_plot, = plt.plot(x, tanh(x), '--' ,label="tanh")

    plt.legend(handles=[relu_plot, sigmoid_plot, tanh_plot])

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Activation Functions")

    plt.savefig('./images/activation_function_overview_dotted.png')

    plt.clf()

def draw_activation_functions(start, stop):

    x = np.arange(start, stop, 0.01)

    relu_plot, = plt.plot(x, ReLU(x),label="ReLU")
    sigmoid_plot, = plt.plot(x, sigmoid(x), label="sigmoid")
    #softmax_plot, = plt.plot(x, softmax(x), label="softmax")
    tanh_plot, = plt.plot(x, tanh(x) ,label="tanh")

    plt.legend(handles=[relu_plot, sigmoid_plot, tanh_plot])

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Activation Functions")

    plt.savefig('./images/activation_function_overview.png')

    plt.clf()


def draw_probabilities_after_activation(start, stop):
    x = np.arange(start, stop, 10)


    names = np.asarray(['A', 'B', 'C', 'D', 'E'])

    heigths = np.asarray([-400, 200, -1200, 800, 1000])
    height_lower_values = heigths / 1000

    height_lower_after_ReLU = ReLU(height_lower_values)
    height_after_ReLU = ReLU(heigths)

    plt.subplot(2, 3, 1)
    plt.bar(names, heigths)
    plt.subplot(2,3,4)
    plt.bar(names, height_lower_values)

    
    plt.subplot(2, 3, 2)
    plt.bar(names, height_after_ReLU)
    plt.subplot(2,3,5)
    plt.bar(names, height_lower_after_ReLU)


    plt.subplot(2,3,3)
    plt.bar(names, softmax(height_after_ReLU))
    plt.subplot(2,3,6)
    plt.bar(names, softmax(height_lower_after_ReLU))


    plt.show()

def draw_softmax_example():

    names = np.asarray(['A', 'B', 'C', 'D', 'E'])

    heigths = np.asarray([-400, 200, -1200, 800, 1000])
    height_lower_values = heigths / 500

    height_lower_after_ReLU = ReLU(height_lower_values)


    plt.subplot(1,3,1)
    plt.bar(names, height_lower_values)
    plt.title("Input", size=12)

    plt.subplot(1,3,2)
    plt.bar(names, height_lower_after_ReLU)
    plt.title("ReLU output",size=12)


    plt.subplot(1,3,3)
    plt.bar(names, softmax(height_lower_after_ReLU))
    plt.title("Softmax output",size=12)

    plt.suptitle('Squashing network output',size=16, y=1.0)

    plt.show()


if __name__ == '__main__':
    #draw_activation_functions_dotted(-4, 4)
    #draw_activation_functions(-4, 4)
    range = 100
    #draw_probabilities_after_activation(0, range)
    draw_softmax_example()