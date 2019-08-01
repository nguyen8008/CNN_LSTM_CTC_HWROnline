# CNN_LSTM_CTC_HWROnline
## Files
* utils_CNN_2BLSTM_CTC.py consists of CNN-2BLSTM-CTC network model.
* train.py is used to train the end-to-end network.
* predict.py is used to evaluate the trained network.

## Guidance
1. Training process using train.py stops after 150 epochs. 
2. Only the best network is kept during training process.
3. After training process, the best network is restored by predict.py to evaluate the trained network.
