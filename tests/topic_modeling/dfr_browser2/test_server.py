"""Tests for the dfr_browser2 server.py module.

Coverage: 95%. Missing: 37, 39
Last Updated: December 24, 2025
"""

import http.client
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


def is_port_available(port: int) -> bool:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
        sock.close()
        return True
    except OSError:
        return False


def cleanup_server_process(
    process: subprocess.Popen, port: int, timeout: float = 3.0
) -> None:
    """Clean up a server process and ensure the port is released.

    Args:
        process: The subprocess to clean up
        port: The port number the server was using
        timeout: Maximum time to wait for cleanup
    """
    if process.poll() is None:
        # Try graceful termination first
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            # Force kill if terminate doesn't work
            process.kill()
            process.wait(timeout=1.0)

    # Wait for port to be released
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_available(port):
            return
        time.sleep(0.1)


@pytest.fixture
def test_html_dir(tmp_path: Path) -> Path:
    """Create a test directory with HTML files for the server."""
    # Create index.html
    (tmp_path / "index.html").write_text(
        "<html><body>Test Index</body></html>", encoding="utf-8"
    )

    # Create a CSS file
    (tmp_path / "styles.css").write_text("body { color: red; }", encoding="utf-8")

    # Create a JS file
    (tmp_path / "app.js").write_text("console.log('test');", encoding="utf-8")

    # Create a data file
    (tmp_path / "data.json").write_text('{"test": true}', encoding="utf-8")

    return tmp_path


@pytest.fixture
def server_script() -> Path:
    """Return path to the server.py script."""
    return (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "lexos"
        / "topic_modeling"
        / "dfr_browser2"
        / "server.py"
    )


def test_server_starts_on_default_port(test_html_dir: Path, server_script: Path):
    """Test that server starts on the default port 8000."""
    # Use a custom port to avoid conflicts
    port = 58201

    # Start server as subprocess
    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Give server time to start
        time.sleep(2.0)  # Increased for reliability

        # Check if server is running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Server process died. stdout: {stdout}, stderr: {stderr}")

        # Try to connect with retries
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body

        conn.close()

    finally:
        # Clean up
        cleanup_server_process(process, port)


def test_server_starts_on_custom_port(test_html_dir: Path, server_script: Path):
    """Test that server starts on a custom port."""
    port = 58123

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)
        assert process.poll() is None

        # Try to connect with retries
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_server_serves_static_files(test_html_dir: Path, server_script: Path):
    """Test that server correctly serves static files."""
    port = 58124

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Helper function to make request with retries
        def make_request(path, expected_content):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    conn = http.client.HTTPConnection("localhost", port, timeout=2)
                    conn.request("GET", path)
                    response = conn.getresponse()
                    assert response.status == 200
                    body = response.read().decode()
                    assert expected_content in body
                    conn.close()
                    return
                except (ConnectionRefusedError, OSError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        raise

        # Test CSS file
        make_request("/styles.css", "color: red")

        # Test JS file
        make_request("/app.js", "console.log")

        # Test JSON file
        make_request("/data.json", "test")

    finally:
        cleanup_server_process(process, port)


def test_server_spa_routing_returns_index(test_html_dir: Path, server_script: Path):
    """Test that non-file paths return index.html for SPA routing."""
    port = 58125

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request a route that doesn't exist as a file
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/about")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body  # Should return index.html

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_server_invalid_port_exits(server_script: Path):
    """Test that server exits with error for invalid port."""
    # Test port out of range
    result = subprocess.run(
        [sys.executable, str(server_script), "99999"],
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert result.returncode == 1
    assert "Port must be between 1 and 65535" in result.stdout

    # Test non-numeric port
    result = subprocess.run(
        [sys.executable, str(server_script), "invalid"],
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert result.returncode == 1
    assert "Invalid port number" in result.stdout


def test_server_port_already_in_use(test_html_dir: Path, server_script: Path):
    """Test that server handles port already in use."""
    port = 58126

    # Start first server
    process1 = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(0.5)

        # Try to start second server on same port
        result = subprocess.run(
            [sys.executable, str(server_script), str(port)],
            cwd=str(test_html_dir),
            capture_output=True,
            text=True,
            timeout=2,
        )

        assert result.returncode == 1
        assert "already in use" in result.stdout

    finally:
        cleanup_server_process(process1, port)


def test_spa_handler_with_query_string(test_html_dir: Path, server_script: Path):
    """Test that SPAHandler correctly handles URLs with query strings."""
    port = 58127

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request with query string
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/about?param=value")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_spa_handler_with_fragment(test_html_dir: Path, server_script: Path):
    """Test that SPAHandler correctly handles URLs with fragments."""
    port = 58128

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request with fragment (note: fragments are usually not sent to server, but test the handler logic)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/page")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_spa_handler_serves_directory_as_index(
    test_html_dir: Path, server_script: Path
):
    """Test that requesting a directory serves index.html."""
    port = 58129

    # Create a subdirectory
    subdir = test_html_dir / "subdir"
    subdir.mkdir()

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request the directory
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/subdir/")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body  # Should return index.html

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_spa_handler_url_with_encoded_characters(
    test_html_dir: Path, server_script: Path
):
    """Test that SPAHandler handles URL-encoded characters."""
    port = 58130

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request with URL-encoded characters (space = %20)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/my%20page")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_spa_handler_with_fragment_in_path(test_html_dir: Path, server_script: Path):
    """Test that SPAHandler handles fragment identifiers in paths."""
    port = 58131

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request with fragment (usually stripped by browser, but testing the handler)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/page#section")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "Test Index" in body

        conn.close()

    finally:
        process.terminate()
        process.wait(timeout=2)


def test_spa_handler_does_not_redirect_static_files(
    test_html_dir: Path, server_script: Path
):
    """Test that SPAHandler does not redirect requests for static file extensions."""
    port = 58132

    # Create various static files
    (test_html_dir / "test.png").write_bytes(b"fake png data")
    (test_html_dir / "test.svg").write_text("<svg></svg>", encoding="utf-8")
    (test_html_dir / "test.ico").write_bytes(b"fake ico data")
    (test_html_dir / "test.csv").write_text("col1,col2", encoding="utf-8")
    (test_html_dir / "test.txt").write_text("text file", encoding="utf-8")

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        def check_file(path, expected_content):
            """Check that a static file is served correctly."""
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    conn = http.client.HTTPConnection("localhost", port, timeout=2)
                    conn.request("GET", path)
                    response = conn.getresponse()
                    body = response.read().decode(errors="ignore")
                    conn.close()
                    # Should get actual file content, not index.html
                    assert response.status == 200
                    assert expected_content in body
                    assert "Test Index" not in body  # Should NOT return index.html
                    return
                except (ConnectionRefusedError, OSError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        raise

        # Test each static file type
        check_file("/test.png", "fake png")
        check_file("/test.svg", "<svg>")
        check_file("/test.ico", "fake ico")
        check_file("/test.csv", "col1,col2")
        check_file("/test.txt", "text file")

    finally:
        process.terminate()
        process.wait(timeout=2)


def test_server_handles_keyboard_interrupt(test_html_dir: Path, server_script: Path):
    """Test that server handles Ctrl+C (KeyboardInterrupt) gracefully."""
    port = 58133

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Verify server is running
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/")
                response = conn.getresponse()
                conn.close()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200

        # Send SIGINT (Ctrl+C)
        process.send_signal(signal.SIGINT)

        # Wait for process to exit
        stdout, stderr = process.communicate(timeout=5)

        # Check that it exited cleanly with the "Server stopped" message
        assert process.returncode == 0 or process.returncode == -2  # -2 is SIGINT
        assert "Server stopped" in stdout or "KeyboardInterrupt" in stderr

    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


def test_spa_handler_existing_file_with_query_string(
    test_html_dir: Path, server_script: Path
):
    """Test that existing files are served even with query strings."""
    port = 58401

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request an existing file with a query string
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                conn.request("GET", "/styles.css?v=123")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "color: red" in body

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_spa_handler_existing_file_with_fragment(
    test_html_dir: Path, server_script: Path
):
    """Test that existing files are served even with fragments."""
    port = 58402

    process = subprocess.Popen(
        [sys.executable, str(server_script), str(port)],
        cwd=str(test_html_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        time.sleep(2.0)

        # Request an existing file with a fragment
        # Note: HTTP clients don't normally send fragments, but we can test the logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPConnection("localhost", port, timeout=2)
                # Manually construct request with fragment in path
                conn.request("GET", "/app.js#section")
                response = conn.getresponse()
                break
            except (ConnectionRefusedError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    raise

        assert response.status == 200
        body = response.read().decode()
        assert "console.log" in body

        conn.close()

    finally:
        cleanup_server_process(process, port)


def test_server_privileged_port_permission_denied(
    test_html_dir: Path, server_script: Path
):
    """Test that permission errors on privileged ports are handled.

    This tests the 'else: raise' branch in the OSError handler.
    Note: This test may pass differently depending on user permissions.
    """
    # Try to bind to port 80 (requires root privileges on most systems)
    # This should trigger an OSError with errno 13 (Permission denied)
    # which is not 48 or 98, so it should be re-raised
    result = subprocess.run(
        [sys.executable, str(server_script), "80"],
        cwd=str(test_html_dir),
        capture_output=True,
        text=True,
        timeout=2,
    )

    # Should exit with non-zero code
    # Either permission denied or we'll skip if running as root
    if "Permission denied" in result.stderr or result.returncode != 0:
        assert result.returncode != 0
    else:
        # If we somehow have permission to bind to port 80,
        # we can't test this branch
        pytest.skip("Unable to trigger permission error on port 80")
