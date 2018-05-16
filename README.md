# Machine learning

This repository is used to experiment and train machine learning models for predictions.
The repository also includes methods for processing InkML files into normalized traces and images.

## Creating dataset and running the project

### Prerequesites
- Python 3.5
- Pip (To python 3.5)

In order to run the code in this repository, you will need either InkML files or an already generated dataset. There is also a complete folder with both xml and preprocessed data available. If this file is chosen, just extract all the files within the ```/online_recog/data``` folder.

First clone the repository and run ```pip install -r requirements.txt ```.

### From InkML files

If you do not have access to the preprocessed dataset. You must first run the preprocessing on InkML files.

1. Download the files from [datasets](https://github.com/bachelor10/recurrent-octo-sniffle/tree/master/datasets)
2. Create folders such that the folder structure in online_recog is:
```/online_recog/data/xml/```
3. Unzip the InkML files downloaded, and paste them into the xml folder.
4. Open ```/online_recog/keras_lstm.py``` in a text editor.
5. Uncomment the line ```generate_and_save_dataset()```
6. Run ```python keras_lstm.py```

### From already preprocessed
If you have downloaded the preprocessed data. You will just have to put the data in the correct folder.

1. Download the files from [datasets](https://github.com/bachelor10/recurrent-octo-sniffle/tree/master/datasets)
2. Create the folder ```/online_recog/data```.
3. Extract the zipped files into the newly created ```/data``` folder.

### Running the training

After placing the datasets into the correct folder, the models can be ran by doing the following.

1. Uncomment```load_dataset_and_run_model()``` in ```/online_recog/keras_lstm_py```. 
2. Choose the model you wish to train by changing ```run_RNN_model()``` to the method with the model you wish to train. 
3. Run ```python keras_lstm.py```.

### Including the real data.

In order to include real data for validation during training. You will either need to create data from symbol-predictor-server, or download the already processed files (the real data is included).

In ```/online_recog/keras_lstm.py```, there is a callback class which can be used to run validation and store logs from the training. This class, as well as a couple other places in ```keras_lstm.py``` has commented out lines that can be uncommented if real data is available in ```/online_recog/data```. The regarding lines is shown in the ```keras_lstm.py``` file.


### Python packages
- [Requirements](requirements.txt)


