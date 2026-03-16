#!/usr/bin/env python3
"""
Simple HTTP server for SPA with HTML5 history routing.
All requests are served index.html to allow client-side routing.

Usage:
    python3 server.py [port]

    port: Optional port number (default: 8000)

Examples:
    python3 server.py          # Run on default port 8000
    python3 server.py 5000     # Run on port 5000
"""

import http.server
import os
import socketserver
import sys
from urllib.parse import unquote

DEFAULT_PORT = 8000


class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Decode URL
        path = unquote(self.path)

        # Remove query string and fragment
        if "?" in path:
            path = path.split("?")[0]
        if "#" in path:
            path = path.split("#")[0]

        # Check if file exists
        file_path = os.path.join(os.getcwd(), path.lstrip("/"))

        # If path is a directory or file doesn't exist, serve index.html
        if os.path.isdir(file_path) or not os.path.exists(file_path):
            # Skip if requesting actual static files
            if not any(
                path.endswith(ext)
                for ext in [
                    ".js",
                    ".css",
                    ".png",
                    ".jpg",
                    ".svg",
                    ".ico",
                    ".json",
                    ".txt",
                    ".csv",
                    ".gz",
                ]
            ):
                self.path = "/index.html"

        return http.server.SimpleHTTPRequestHandler.do_GET(self)


if __name__ == "__main__":
    # Parse command-line arguments for port number
    port = DEFAULT_PORT

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1 or port > 65535:
                print("Error: Port must be between 1 and 65535")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid port number '{sys.argv[1]}'")
            print("Usage: python3 server.py [port]")
            sys.exit(1)

    try:
        with socketserver.TCPServer(("", port), SPAHandler) as httpd:
            print(f"Server running at http://localhost:{port}/")
            print("Press Ctrl+C to stop")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped")
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"Error: Port {port} is already in use")
            print("Try a different port: python3 server.py <port>")
            sys.exit(1)
        else:
            raise
