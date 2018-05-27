# MNIST web
web site for testing neural networks on MNIST data set using 2 hidden layers

### To run this you need
* node.js
* pytorch

### Instructions
1. run `node --version` on your cmd, if not found go to [here](https://nodejs.org/en/) and download node.js.

2. check if you have pytorch installed with this script
```python
import torch
print(torch.__version__)
```

3. run the server by typing `node server.js` in the cmd.

4. your server is ready for requests(*UI in progress*) send a http post request with JSON following the syntex below.

### Syntex
##### General Settings
`neural_net` - the neural net you want to use out of {"Basic", "Dropout", "Batch_norm", "Combine"}
`epochs` - number of passes on all the data
`learning_rate` - learning rate of the network
`batch_size` - passing on data using *batch_size* number of samples each iteration
`valid_split` - split your train data to validation and training and evalute the network each epoch

##### Structure Settings
`hidden1_size` - number of neurons on the first hidden layer
`hidden2_size` - number of neurons on the second hidden layer

##### Options
`write_test_pred` - write predictions to file 'test.pred', **number** to represent a boolean(_due to a bug in passing a boolean from node.js to python_) {0, 1}
`draw_loss_graph` - graph loss of training and validation, **number** to represent a boolean(_due to a bug in passing a boolean from node.js to python_) {0, 1}
