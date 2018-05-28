console.log('server is starting...')

const express = require('express')
var bodyParser = require('body-parser');
var chalk = require('chalk');
var pythonShell = require('python-shell');
const hbs = require('express-handlebars').create({ defaultLayout: 'main.hbs' });

const app = express();
app.engine('hbs', hbs.engine);
app.set('view engine', 'hbs');

var port = 8000;

// app.use(express.static('website/form.html'))
app.use(bodyParser.json());


app.get('/', function (req, res) {
    console.log('client get request: ')
    console.log(req.query)
    res.render('form', { title: 'Network calculation request' });
});

// TODO right now form.hbs is main -- make index.hbs main page with link to form.hbs
// app.get('/Calculate', function (req, res) {

// });

app.post('/', function (req, res) {
    console.log('client post request')

    var neural_net = req.body.neural_net
    var epochs = req.body.epochs
    var learning_rate = req.body.learning_rate
    var batch_size = req.body.batch_size
    var valid_split = req.body.valid_split
    var hidden1_size = req.body.hidden1_size
    var hidden2_size = req.body.hidden2_size
    var write_test_pred = req.body.write_test_pred
    var draw_loss_graph = req.body.draw_loss_graph

    var options = {
        args: ['-n', neural_net, '-e', epochs, '-l', learning_rate, '-b', batch_size, '-s', valid_split,
            '-h1', hidden1_size, '-h2', hidden2_size, '-w', write_test_pred, '-d', draw_loss_graph]
    }

    pythonShell.run('MNIST_script.py', options, function (err, results) {
        if (err) res.end(err.message);
        else res.end(results.toString());
    });
})


app.use(function (error, request, response, next) {

    console.log(chalk.red.bold("ERROR"));
    console.log(chalk.red.bold("==========="));
    console.log(error);

    if (!response.headersSent) {

        response
            .status(500)
            .send("Sorry - something went wrong...")
            ;

    }

}
);

// start server
app.listen(port, function () {
    console.log('Server started At http://localhost:' + port);
});
