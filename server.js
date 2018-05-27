console.log('server is starting...')

var express = require('express')
var bodyParser = require('body-parser');

var app = express()

var server = app.listen(8000, listening)

function listening() {
    console.log('listening...')
}

app.use(express.static('website'))
app.use(bodyParser.json());


let runPy = new Promise(function (success, nosuccess) {

    const { spawn } = require('child_process');
    const pyprog = spawn('python', ['./MNIST_script.py']);

    pyprog.stdout.on('data', function (data) {

        success(data);

    });
    pyprog.stderr.on('data', (data) => {

        nosuccess(data);

    });

});

app.post('/', function (req, res) {
    var neural_net = req.body.neural_net
    // var epochs = req.body.epochs
    // var learning_rate = req.body.learning_rate
    // var batch_size = req.body.batch_size
    // var valid_split = req.body.valid_split
    // var hidden1_size = req.body.hidden1_size
    // var hidden2_size = req.body.hidden2_size
    // var write_test_pred = req.body.write_test_pred
    // var draw_loss_graph = req.body.draw_loss_graph

    runPy.then(function (fromRunpy) {
        res.end(fromRunpy);
    });
})