"""
Local static server for the YOLO26 multi-task webcam app.

Sends the headers that make the page "cross-origin isolated", which ONNX Runtime Web
needs for multi-threaded WebAssembly (SharedArrayBuffer):
    Cross-Origin-Opener-Policy: same-origin
    Cross-Origin-Embedder-Policy: require-corp

Only uses the Python standard library - works on Windows and macOS:
    Windows:  python serve.py
    macOS:    python3 serve.py
Then open http://localhost:8000  (camera access needs localhost or https).
Optional:  python serve.py --port 8080
"""
import argparse
import functools
import http.server
import os
import socket
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))


class IsolatedHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'text/javascript',
        '.mjs': 'text/javascript',
        '.wasm': 'application/wasm',
        '.onnx': 'application/octet-stream',
        '.json': 'application/json',
    }

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        # always serve fresh files while developing (models are cached by the app itself)
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def log_message(self, fmt, *args):
        # keep the console readable: only log errors
        if len(args) > 1 and str(args[1]).startswith(('4', '5')):
            super().log_message(fmt, *args)


def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(('127.0.0.1', port)) != 0


def main():
    parser = argparse.ArgumentParser(description='Serve the YOLO26 webcam app with COOP/COEP headers.')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true', help='do not open the browser automatically')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(ROOT, 'models', 'manifest.json')):
        print('WARNING: models/manifest.json not found - run export_models.py first.')
    if not port_free(args.port):
        print(f'ERROR: port {args.port} is already in use. Close the other server or use --port 8080')
        sys.exit(1)

    handler = functools.partial(IsolatedHandler, directory=ROOT)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    url = f'http://localhost:{args.port}/'
    print('=' * 60)
    print(f'  YOLO26 webcam app running at {url}')
    print('  (COOP/COEP enabled -> multi-threaded WebAssembly)')
    print('  Press Ctrl+C to stop.')
    print('=' * 60)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
