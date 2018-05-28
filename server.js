console.log('server is starting...')

const express = require('express')
var bodyParser = require('body-parser');
var chalk = require('chalk');
var pythonShell = require('python-shell');
var morgan = require('morgan');
const hbs = require('express-handlebars').create({ defaultLayout: 'main.hbs' });
const { parse } = require('querystring');

const app = express();
app.engine('hbs', hbs.engine);
app.set('view engine', 'hbs');

var port = 8000;

app.use(morgan('common'))
app.use(express.static(__dirname + '/assets'))
app.use(bodyParser.json());


app.get('/', function (req, res) {
    res.render('form', { title: 'Network calculation request' });
});

app.get('/about', function (request, response) {
    response.render('about', { title: 'About' });
});

// TODO right now form.hbs is main -- make index.hbs main page with link to form.hbs
// app.get('/Calculate', function (req, res) {

// });

app.post('/', function (req, res) {

    let body = '';
    let data = '';
    req.on('data', chunk => {
        body += chunk.toString();
    });
    req.on('end', () => {
        console.log(parse(body));
        data = parse(body)

        var neural_net = data.neural_net
        var epochs = data.epochs
        var learning_rate = data.learning_rate
        var batch_size = data.batch_size
        var valid_split = data.valid_split
        var hidden1_size = data.hidden1_size
        var hidden2_size = data.hidden2_size
        var write_test_pred = data.write_test_pred
        var draw_loss_graph = data.draw_loss_graph

        var options = {
            args: ['-n', neural_net, '-e', epochs, '-l', learning_rate, '-b', batch_size, '-s', valid_split,
                '-h1', hidden1_size, '-h2', hidden2_size, '-w', write_test_pred, '-d', draw_loss_graph]
        }

        pythonShell.run('MNIST_script.py', options, function (err, results) {
            if (err) res.end(err.message);
            else res.end(results.toString());
        });
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
