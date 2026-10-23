const LOOPBACK_HOST = "127.0.0.1";

function listenOnLoopback(app, port, callback) {
    return app.listen(port, LOOPBACK_HOST, callback);
}

module.exports = { listenOnLoopback, LOOPBACK_HOST };
