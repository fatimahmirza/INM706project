# Comparitive analysis of 1D CNN and BERT models for sentiment analysis
## Overview 
The content of this repository related to the coursework of INM706. This readme file provides the detailed instruction on how to set-up and training procedure for the baseline model and also for the improved model. An implementation of "1D CNN"  and "BERT" for advanced model for sentiment analysis.
## Dataset 
you can instead download everything from that shared link since the dataset is included as well as all the code files.  
https://drive.google.com/drive/folders/1YsUV7jVLPoi25IZnitFLKpVbK2GyqklZ?usp=drive_link
## Prerequisistes
Before running the code, ensure you have sucessfully installed the following files
- requirement.txt
- pip install -r requirement.txt
- Python 3.8
- GPU + CUDA
## Setup
### Configuration 
Set up the configuration file first. You need to put your API key in the file if you want to load the weights and biases.
- config = load_config("config.yaml")
- print(config)
- LEARNING_RATE = float(config['models_config']['baseline']['learning_rate'])
- BATCH_SIZE = int(config['models_config']['batch_size'])
- WEIGHT_DECAY = int(config['models_config']['weight_decay'])
- EPOCHS = int(config['models_config']['epochs'])
- - API_KEY = str(config['models_config']['API_KEY'])
- CHECKPOINT_MODEL = float(config['models_config']['checkpoint_mAP'])

- os.environ["WANDB_API_KEY"] = API_KEY
- Similarly for the improved model as well.
### Load model
Load the pretrained modle from the file 
- LOAD_MODEL_FILE= r'saved_models/model_full_data_80_tensor(0.9161).pth.tar' : put your path directory here.
### Running or Testing
For further development or testing you need to run it locally.You need to put your path directory in the file "train_1d_cnn"for 1D CNN model 
and in the file " train_bert" for bert.
### Train models
- To train 1D CNN model put your path directories in the file where "text_data/train.csv" 
- To train BERT model put your path directories in the file where "text_data/train.csv"
## Hyperperameters
# Hyperparameters
-vocab_size = len(top_words)
-embedding_dim = cnn1d_config['embedding_dim']
-output_dim = cnn1d_config['output_dim']
-kernel_size = cnn1d_config['kernel_size']
-stride = cnn1d_config['stride']
-num_epochs = cnn1d_config['num_epochs']
-batch_size = cnn1d_config['batch_size']
-learning_rate = cnn1d_config['lr']
## Checkpoints 
-# Save model after training
-torch.save(model.state_dict(), "models/1d_cnn.pth")
wandb.save()
## Results
-The baseline 1D CNN model achieved training accuracy of around 45%, and testing accuracy of around 40%.
-In contrast, the BERT model achieved a high training accuracy of 88% and maintaining a testing accuracy of 79%, nearly 39% higher on test set of data.


