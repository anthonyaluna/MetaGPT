function rejectBrowserOrigin(req, res, next) {
    if (req.headers.origin !== undefined) {
        res.statusCode = 403;
        res.end("Browser-origin requests are not allowed.");
        return;
    }

    next();
}

module.exports = { rejectBrowserOrigin };
