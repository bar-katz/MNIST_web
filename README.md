# MNIST web
A web site for testing neural networks on MNIST data set using 2 hidden layers.

### Required Software
* Node.js
* PyTorch

### Instructions
1. Run `node --version` on your cmd. If not found, go to [here](https://nodejs.org/en/) and download Node.js.

2. Check if you have PyTorch installed with this script:
    ```python
    import torch
    print(torch.__version__)
    ```

3. Run the server by typing `node server.js` in the cmd.

4. Your server is now ready for requests (*UI in progress*). Send an HTTP POST request via JSON, following the syntax below.

### Syntax
##### General Settings
- `neural_net` - The neural net you want to use out of {"Basic", "Dropout", "Batch_norm", "Combine"}.
- `epochs` - The number of data iterations.
- `learning_rate` - The network's learning rate {_small values like_ 0.01, 0.005, 0.001}.
- `batch_size` - Iterating the data using *batch_size* number of samples each iteration {normally 64}
- `valid_split` - Splits your train data to validation and training and evalutes the network after each epoch {ranges from 0 to 1}.

##### Structure Settings
- `hidden1_size` - The number of neurons in the first hidden layer {default is 100}.
- `hidden2_size` - The number of neurons in the second hidden layer {default is 50}.

##### Options
- `write_test_pred` - Writes predictions to file 'test.pred', **number** to represent a boolean(_due to a bug in passing a boolean from node.js to python_) {0, 1}.
- `draw_loss_graph` - Draws loss graph of training and validation, **number** to represent a boolean(_due to a bug in passing a boolean from node.js to python_) {0, 1}.
