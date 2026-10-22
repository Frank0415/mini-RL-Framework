import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    assets = Path.home() / ".cache/mini-rl/perfetto-v58.2"
    if not (assets / "index.html").exists():
        with urlopen("https://github.com/google/perfetto/releases/download/v58.2/perfetto-ui.zip") as response:
            with ZipFile(BytesIO(response.read())) as archive:
                archive.extractall(assets)
    static = Path(__file__).with_name("static")
    routes = {
        "/rl/": static / "index.html",
        "/rl/app.js": static / "app.js",
        "/rl/style.css": static / "style.css",
        "/rl/trace.json": args.trace.resolve(),
    }

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            return str(routes.get(urlsplit(path).path, super().translate_path(path)))

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, format, *args):
            pass

    handler = partial(Handler, directory=str(assets))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"RL Trace: http://127.0.0.1:{args.port}/rl/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
