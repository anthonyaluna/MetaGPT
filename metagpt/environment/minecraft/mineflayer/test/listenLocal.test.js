const assert = require("assert");
const http = require("http");
const { listenOnLoopback, LOOPBACK_HOST } = require("../lib/listenLocal");
const { rejectBrowserOrigin } = require("../lib/rejectBrowserOrigin");

function requestLocal(port, method, headers = {}) {
    return new Promise((resolve, reject) => {
        const request = http.request(
            { host: LOOPBACK_HOST, method, path: "/", port, headers },
            (response) => {
                let body = "";
                response.setEncoding("utf8");
                response.on("data", (chunk) => {
                    body += chunk;
                });
                response.on("end", () => {
                    resolve({ statusCode: response.statusCode, body });
                });
            },
        );
        request.on("error", reject);
        request.end();
    });
}

async function main() {
    const server = http.createServer((request, response) => {
        rejectBrowserOrigin(request, response, () => {
            response.end("local-ok");
        });
    });

    try {
        await new Promise((resolve, reject) => {
            server.once("error", reject);
            listenOnLoopback(server, 0, resolve);
        });

        const address = server.address();
        assert.ok(address && typeof address === "object");
        assert.strictEqual(address.address, LOOPBACK_HOST);

        const response = await requestLocal(address.port, "GET");
        assert.strictEqual(response.statusCode, 200);
        assert.strictEqual(response.body, "local-ok");

        const browserResponse = await requestLocal(address.port, "POST", {
            Origin: "https://attacker.example",
        });
        assert.strictEqual(browserResponse.statusCode, 403);
        assert.strictEqual(browserResponse.body, "Browser-origin requests are not allowed.");
        console.log("Mineflayer listener binds to IPv4 loopback and rejects browser-origin requests.");
    } finally {
        await new Promise((resolve, reject) => {
            server.close((error) => (error ? reject(error) : resolve()));
        });
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
