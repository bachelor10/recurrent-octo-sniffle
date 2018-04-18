import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def read_log_files(name="combined_model"):
    filename = "../online_recog/logs/" + name + "_"

    real_data_acc = np.load(filename + "real_data_acc.npy")
    real_data_loss = np.load(filename + "real_data_loss.npy")
    
    train_data_acc = np.load(filename + "train_data_acc.npy")
    train_data_loss = np.load(filename + "train_data_loss.npy")

    validation_data_acc = np.load(filename + "validation_data_acc.npy")
    validation_data_loss = np.load(filename + "validation_data_loss.npy")

    return train_data_acc, train_data_loss, validation_data_acc, validation_data_loss, real_data_acc, real_data_loss

def vizualize_epoch_logs(acc, loss, val_acc, val_loss, real_acc, real_loss, title):
    plt.plot(acc)
    plt.plot(val_acc)
    plt.plot(real_acc)
    plt.ylim([0.3, 1])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test', 'real'], loc='upper left')
    plt.savefig('./images/' + title)




acc, loss, val_acc, val_loss, real_acc, real_loss = read_log_files()
acc1, loss1, val_acc1, val_loss1, real_acc1, real_loss1 = read_log_files(name="CNN_model")
acc2, loss2, val_acc2, val_loss2, real_acc2, real_loss2 = read_log_files(name="RNN_model")

vizualize_epoch_logs(acc, loss, val_acc, val_loss, real_acc, real_loss, "Combined CNN and RNN (5000)")
plt.figure()
vizualize_epoch_logs(acc1, loss1, val_acc1, val_loss1, real_acc1, real_loss1, "CNN (5000)")
plt.figure()
vizualize_epoch_logs(acc2, loss2, val_acc2, val_loss2, real_acc2, real_loss2, "RNN (5000)")
plt.show()