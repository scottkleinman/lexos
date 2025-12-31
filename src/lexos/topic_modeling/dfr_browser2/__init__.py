"""__init__.py.

Last Updated: December 25, 2025
Last Tested: December 25, 2025
"""

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from .prepare_data import process_mallet_state_file


class Browser(BaseModel):
    """Browser class to create and serve DFR Browser 2.

    filename_map usage:
    - Provide a mapping of `original_filename` -> `destination_filename` where
      `original_filename` is the filename present in `mallet_files_path` and
      `destination_filename` is the name to use under the browser's `data/` folder.

    Example:
        filename_map = {"doc-topics.txt": "doc-topic.txt"}
        Browser(..., filename_map=filename_map)
    """

    mallet_files_path: str = Field(
        ..., description="Path to the folder containing Mallet output files."
    )
    browser_path: str = Field(
        None, description="The folder where the browser will be saved."
    )
    template_path: str = Field(
        "dist", description="Path to the DFR Browser 2 template folder."
    )
    data_path: str | None = Field(
        None,
        description=(
            "Path to a tab-separated (TSV) file containing the original data used "
            "to generate the topic model. Each row must contain 2 or 3 columns. "
            "This file will be copied into the browser's data folder."
        ),
    )
    config: dict | None = Field(
        None, description="Configuration dictionary for the DFR Browser 2."
    )
    port: int = Field(8000, description="Port number for serving the DFR Browser 2.")
    filename_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of original filenames to new filenames.",
    )
    copied_files: dict = Field(
        default_factory=dict, description="Tracks copied files for config updates."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Version of the browser distribution this class is creating/serving
    BROWSER_VERSION: ClassVar[str] = "0.2.3"

    # Class-level constants for required files
    REQUIRED_FILES: ClassVar[set[str]] = {
        "metadata.csv",
        "topic-keys.txt",
        "doc-topic.txt",
        "topic-state.gz",
        "topic_coords.csv",
    }

    ALT_NAME_GROUPS: ClassVar[dict[str, list[str]]] = {
        "doc-topic.txt": ["doc-topics.txt", "doc-topic.txt"],
        "topic-state.gz": ["topic-state.gz", "state.gz"],
    }

    @property
    def version(self):
        """Return the version of the DFR Browser."""
        # If the user or template supplies a version, prefer that value
        try:
            if self.config and isinstance(self.config, dict):
                app_cfg = self.config.get("application") or {}
                if isinstance(app_cfg, dict) and app_cfg.get("version"):
                    return app_cfg.get("version")
        except Exception:
            # defensive fallback
            pass
        return self.BROWSER_VERSION

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept assignments to `config` and write merged config to disk.

        This ensures that once `self.config` is set, the persisted `config.json`
        file will be updated to match the merged configuration. We only write
        after the instance is fully initialized to avoid partial writes during
        construction.

        Args:
            name (str): Attribute name.
            value (Any): Attribute value.
        """
        # Use BaseModel's setattr to set attribute (ensures pydantic behavior)
        super().__setattr__(name, value)
        # Only write to disk after init and when config is updated
        if name == "config" and getattr(self, "_initialized", False):
            try:
                self._write_config()
            except Exception:
                # Defensive: don't raise during attribute setting
                pass

    def __init__(self, **data: Any) -> None:
        """Initialize the DFR Browser 2 class.

        Args:
            **data (Any): Keyword arguments for the BaseModel.
        """
        # First call BaseModel initializer
        super().__init__(**data)

        # Initialize private attribute for server process (not a Pydantic field)
        object.__setattr__(self, "_server_process", None)

        # Execute initialization steps
        self._validate_and_setup_paths()
        self._check_and_prepare_required_files()
        self._copy_template()
        self._copy_data_file()
        self._copy_mallet_files()
        self._write_config()

        # Mark initialization complete
        object.__setattr__(self, "_initialized", True)

    def _validate_and_setup_paths(self) -> None:
        """Validate and setup file paths for the browser."""
        # Convert paths into Path objects
        self.mallet_files_path = Path(self.mallet_files_path)

        # If template_path is the default "dist", resolve it relative to this module
        if self.template_path == "dist":
            module_dir = Path(__file__).parent
            self.template_path = module_dir / "dist"
        else:
            self.template_path = Path(self.template_path)

        if self.browser_path:
            self.browser_path = Path(self.browser_path)
        else:
            # Create temp directory if none provided
            self.browser_path = Path(tempfile.mkdtemp(prefix="dfr_browser_"))

        # Check the template and mallet path exist
        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Template path does not exist: {self.template_path}"
            )
        if not self.mallet_files_path.exists():
            raise FileNotFoundError(
                f"Mallet files path does not exist: {self.mallet_files_path}"
            )

    def _check_file_exists_with_alternates(
        self, canonical_name: str
    ) -> tuple[bool, Path | None]:
        """Check if a file exists, considering alternate names.

        Args:
            canonical_name: The canonical name of the file to check.

        Returns:
            tuple: (exists: bool, path: Path | None) where path is the actual file path if found.
        """
        # Check if this file has alternate names
        if canonical_name in self.ALT_NAME_GROUPS:
            for alt in self.ALT_NAME_GROUPS[canonical_name]:
                if (self.mallet_files_path / alt).exists():
                    return True, self.mallet_files_path / alt
        else:
            # Check the canonical name directly
            if (self.mallet_files_path / canonical_name).exists():
                return True, self.mallet_files_path / canonical_name

        # Check filename_map
        if self.filename_map:
            for alt in self.ALT_NAME_GROUPS.get(canonical_name, [canonical_name]):
                if alt in self.filename_map:
                    mapped_path = self.mallet_files_path / self.filename_map[alt]
                    if mapped_path.exists():
                        return True, mapped_path

        return False, None

    def _check_and_prepare_required_files(self) -> None:
        """Check required files exist and generate topic_coords.csv if missing."""
        missing_files = []
        missing_topic_coords = False

        for required_file in self.REQUIRED_FILES:
            exists, _ = self._check_file_exists_with_alternates(required_file)
            if not exists:
                if required_file == "topic_coords.csv":
                    missing_topic_coords = True
                else:
                    missing_files.append(required_file)

        # If topic_coords.csv is missing but topic-state.gz exists, generate it
        if missing_topic_coords:
            state_exists, state_path = self._check_file_exists_with_alternates(
                "topic-state.gz"
            )
            if state_exists:
                print(
                    f"topic_coords.csv not found. Generating from {state_path.name}..."
                )
                try:
                    process_mallet_state_file(
                        state_file=str(state_path),
                        output_dir=str(self.mallet_files_path),
                        n_top_words=30,
                        generate_all=False,
                    )
                    print("Successfully generated topic_coords.csv")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to generate topic_coords.csv from topic-state file: {e}"
                    ) from e
            else:
                missing_files.append("topic_coords.csv")

        # Raise exception if any required files are still missing
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {self.mallet_files_path}: {', '.join(missing_files)}"
            )

    def _copy_template(self) -> None:
        """Copy the browser template to the browser_path."""
        # Determine filenames to check. If user provided filename_map, treat
        # keys as filenames that exist in the mallet files path (orig names),
        # and values as the desired destination names.
        filenames_to_check = set(self.REQUIRED_FILES)
        if self.filename_map:
            # Include keys from the filename_map (original filenames)
            filenames_to_check.update(self.filename_map.keys())

        # Store for use in _copy_mallet_files
        self._filenames_to_check = filenames_to_check

        try:
            shutil.copytree(self.template_path, self.browser_path, dirs_exist_ok=True)
        except Exception:
            # On some systems, copying into existing directory fails for file metadata; fallback to per-file copy
            for src in self.template_path.rglob("*"):
                rel = src.relative_to(self.template_path)
                dest = self.browser_path / rel
                if src.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    shutil.copy2(src, dest)

    def _copy_data_file(self) -> None:
        """Copy the data file to the browser's data directory if provided."""
        if not self.data_path:
            return

        if self.data_path:
            data_src = Path(self.data_path)
            if not data_src.exists():
                raise FileNotFoundError(f"data_path does not exist: {data_src}")
            if data_src.is_dir():
                raise ValueError(
                    "data_path must be a path to a TSV file, not a directory"
                )

            # Validate TSV structure: each non-empty row must have 2 or 3 columns
            with open(data_src, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    cols = line.split("\t")
                    if len(cols) not in (2, 3):
                        raise ValueError(
                            f"Invalid TSV format in {data_src} at line {i}: expected 2 or 3 columns, got {len(cols)}"
                        )

            data_target = self.browser_path / "data"
            data_target.mkdir(parents=True, exist_ok=True)
            # Copy file to the data directory using the expected filename 'docs.txt'
            docs_filename = "docs.txt"
            shutil.copy2(data_src, data_target / docs_filename)
            # Track which files we copied (so we can update config.json paths)
            self.copied_files = getattr(self, "copied_files", {})
            # Save relative path as used by the template
            self.copied_files["data_source"] = f"data/{docs_filename}"

    def _copy_mallet_files(self) -> None:
        """Copy MALLET output files to the browser's data directory."""
        # Copy mallet files into the 'data' folder (not a mallet subfolder)
        data_target = self.browser_path / "data"
        data_target.mkdir(parents=True, exist_ok=True)
        # Ensure we have a holder for copied files metadata
        self.copied_files = getattr(self, "copied_files", {})
        copied_destnames = set()
        # Expand filenames to include alternate names so we can copy the actual file
        filenames_to_check = self._filenames_to_check
        copy_candidates = set(filenames_to_check)
        for canonical, alts in self.ALT_NAME_GROUPS.items():
            for alt in alts:
                copy_candidates.add(alt)
        for src_name in copy_candidates:
            # If the src_name is a filename that only appears as a mapping value
            # (i.e., the mapping is reversed where key is the destination), then
            # skip copying this source directly — the mapping key will handle copying/rename.
            if (
                self.filename_map
                and src_name in set(self.filename_map.values())
                and (src_name not in self.filename_map)
            ):
                continue
            # Interpret src_name as the original filename (key)
            src_path = self.mallet_files_path / src_name
            dest_filename = src_name
            # If the filename is an alternate name for a canonical file, default to the canonical
            # filename as the destination unless the user explicitly specified a mapping for it.
            if not (self.filename_map and src_name in self.filename_map):
                for canonical, alts in self.ALT_NAME_GROUPS.items():
                    if src_name in alts:
                        dest_filename = canonical
                        break
            # If a mapping exists for this src_name, the mapped value is the destination filename
            if self.filename_map and src_name in self.filename_map:
                dest_filename = self.filename_map[src_name]
            # If the file doesn't exist under the original name, try the destination name
            if not src_path.exists():
                fallback_path = self.mallet_files_path / dest_filename
                if fallback_path.exists():
                    src_path = fallback_path
                    # If the user supplied a mapping where the key was actually the
                    # desired destination (i.e., mapping was reversed), then swap
                    # the dest filename so we rename original->dest correctly.
                    if (
                        self.filename_map
                        and src_name in self.filename_map
                        and (self.filename_map[src_name] == fallback_path.name)
                    ):
                        # key was destination, value was source, so swap
                        dest_filename = src_name
                else:
                    # If missing, skip
                    continue
            # Prevent duplicate destination filenames (e.g. doc-topic vs doc-topics)
            if dest_filename in copied_destnames:
                continue
            dest_path = data_target / dest_filename
            shutil.copy2(src_path, dest_path)
            copied_destnames.add(dest_filename)

            # Track copied files for config
            self._track_copied_file(src_name, dest_filename)

        # After copying required files, scan for and copy optional MALLET output files
        optional_files = ["diagnostics.xml"]
        for optional_file in optional_files:
            src_path = self.mallet_files_path / optional_file
            if src_path.exists() and optional_file not in copied_destnames:
                dest_path = data_target / optional_file
                shutil.copy2(src_path, dest_path)
                copied_destnames.add(optional_file)
                # Track for config
                self._track_copied_file(optional_file, optional_file)

    def _track_copied_file(self, src_name: str, dest_filename: str) -> None:
        """Track a copied file for config.json updates.

        Args:
            src_name: Original source filename
            dest_filename: Destination filename
        """
        # Determine canonical group for src_name
        canonical_for_src = None
        for canonical, alts in self.ALT_NAME_GROUPS.items():
            if src_name in alts:
                canonical_for_src = canonical
                break

        lower = dest_filename.lower()
        if canonical_for_src == "doc-topic.txt":
            # If source belonged to doc-topic alt group, map regardless of dest filename
            self.copied_files["doc_topic_file"] = f"data/{dest_filename}"
        if "topic-keys" in lower or "topic_keys" in lower:
            self.copied_files["topic_keys_file"] = f"data/{dest_filename}"
        elif "doc-topic" in lower or "doc-topics" in lower or "doc_topic" in lower:
            # Template uses 'doc_topic_file'
            self.copied_files["doc_topic_file"] = f"data/{dest_filename}"
        elif canonical_for_src == "topic-state.gz":
            # 'topic-state.gz' canonical group — ensure mapping written even if dest filename doesn't contain keyword
            self.copied_files["topic_state_file"] = f"data/{dest_filename}"
        elif "metadata" in lower:
            self.copied_files["metadata_file"] = f"data/{dest_filename}"
        elif "topic-state" in lower or "topic_state" in lower:
            self.copied_files["topic_state_file"] = f"data/{dest_filename}"
        elif "topic_coords" in lower or "topic-coords" in lower:
            self.copied_files["topic_coords_file"] = f"data/{dest_filename}"
        elif "diagnostics" in lower:
            self.copied_files["diagnostics_file"] = f"data/{dest_filename}"

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for use.

        Args:
            port: Port number to check.

        Returns:
            bool: True if port is available, False if already in use.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return True
        except OSError:
            return False

    def config_browser(self, config: dict) -> None:
        """Set the browser configuration after initialization.

        Args:
            config (dict): Configuration dictionary for the DFR Browser 2.
        """
        # Update the config attribute
        self.config = config
        # Write the new config to config.json in browser_path
        self._write_config()

    def serve(self) -> None:
        """Serve the DFR Browser 2 using the server.py subprocess.

        This method starts the server as a subprocess and provides instructions
        for stopping it. Works in both command line and Jupyter notebook environments.
        """
        if not self.browser_path.exists():
            raise FileNotFoundError(f"Browser path does not exist: {self.browser_path}")

        # Check if port is already in use
        if not self._is_port_available(self.port):
            error_msg = (
                f"Port {self.port} is already in use.\n\n"
                f"To resolve this issue, you can:\n"
                f"  1. Stop the process using the port:\n"
                f"     - Check what's using it: lsof -i:{self.port}\n"
                f"     - Kill the process: kill $(lsof -t -i:{self.port})\n"
                f"     - Or in Jupyter: !kill $(lsof -t -i:{self.port})\n"
                f"  2. Use a different port by setting port=<number> when creating the Browser\n"
            )
            raise RuntimeError(error_msg)

        # Find the server.py script in the same directory as this module
        server_script = Path(__file__).parent / "server.py"
        if not server_script.exists():
            raise FileNotFoundError(f"server.py not found at {server_script}")

        # Build the command to run the server
        cmd = [sys.executable, str(server_script), str(self.port)]

        url = f"http://localhost:{self.port}/"

        # Detect if we're running in a Jupyter notebook
        try:
            # Check for IPython/Jupyter environment
            get_ipython()  # type: ignore
            in_notebook = True
        except NameError:
            in_notebook = False

        if in_notebook:
            # Jupyter notebook instructions
            print(f"Starting DFR Browser server at {url}")
            print("\n" + "=" * 60)
            print("To stop the server in Jupyter, run:")
            print("  b.stop_server()")
            print("or")
            print(f"  !kill $(lsof -t -i:{self.port})")
            print("or restart the kernel")
            print("=" * 60 + "\n")

            # Start server as background process
            process = subprocess.Popen(
                cmd,
                cwd=str(self.browser_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Store the process so we can stop it later
            self._server_process = process

            # Give it a moment to start
            import time

            time.sleep(1)

            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                error_msg = f"Server failed to start.\n\nError details:\n{stderr}"
                if "Address already in use" in stderr or "Errno 98" in stderr:
                    error_msg += (
                        f"\n\nPort {self.port} appears to be in use.\n"
                        f"Kill the process with: !kill $(lsof -t -i:{self.port})"
                    )
                raise RuntimeError(error_msg)

            # Open browser
            try:
                webbrowser.open(url)
            except Exception:
                print(f"Please open {url} in your browser")

            print(f"Server running (PID: {process.pid})")

        else:
            # Command line - run with subprocess and wait
            print(f"Starting DFR Browser server at {url}")
            print("Press Ctrl+C to stop the server\n")

            # Open browser before starting server
            try:
                webbrowser.open(url)
            except Exception:
                print(f"Please open {url} in your browser")

            try:
                # Run server and wait (blocks until interrupted)
                subprocess.run(cmd, cwd=str(self.browser_path), check=True)
            except KeyboardInterrupt:
                print("\nServer stopped")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Server error: {e}")

    def stop_server(self) -> None:
        """Stop the running server process.

        This method terminates the server subprocess if it's running in background mode
        (typically in Jupyter notebooks). If the server is not running or was started in
        blocking mode (command line), this method does nothing.

        Example:
            >>> b = Browser(...)
            >>> b.serve()  # Starts server in background
            >>> # ... do some work ...
            >>> b.stop_server()  # Stops the server
        """
        if self._server_process is None:
            print(
                "No server process to stop (server may not have been started or was started in blocking mode)"
            )
            return

        if self._server_process.poll() is not None:
            # Process already terminated
            print("Server process has already stopped")
            self._server_process = None
            return

        try:
            # Terminate the process
            self._server_process.terminate()

            # Wait for process to terminate (with timeout)
            try:
                self._server_process.wait(timeout=5)
                print(f"Server stopped (PID: {self._server_process.pid})")
            except subprocess.TimeoutExpired:
                # If it doesn't terminate gracefully, force kill
                self._server_process.kill()
                self._server_process.wait()
                print(f"Server forcefully stopped (PID: {self._server_process.pid})")
        except Exception as e:
            print(f"Error stopping server: {e}")
        finally:
            self._server_process = None

    def _write_config(self) -> None:
        """Write or update config.json in the browser_path."""
        if not self.browser_path:
            raise ValueError("browser_path is not set")
        self.browser_path.mkdir(parents=True, exist_ok=True)
        cfg_path = Path(self.browser_path) / "config.json"
        # Load existing template config.json (if it exists) as the base
        base_cfg = {}
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    base_cfg = json.load(fh)
            except Exception:
                base_cfg = {}

        # Merge user-provided config into base (user overrides base)
        merged_cfg = dict(base_cfg)
        if self.config:
            merged_cfg.update(self.config)

        # Ensure file paths for known data files point to the data/ folder
        copied = getattr(self, "copied_files", {}) or {}
        for key, rel_path in copied.items():
            # File path precedence:
            # 1. User-specified config (self.config) — should win and be preserved
            # 2. Copied file paths (this process) — override template values
            # 3. Template defaults — only used if no user or copied path applies.
            # If the user explicitly provided this key in the config, preserve it.
            if self.config and key in self.config:
                continue
            # Otherwise, set the key to the path of the copied file (overriding template)
            merged_cfg[key] = rel_path

        # Ensure the application version is present in the config. Preserve user-provided values.
        if "application" not in merged_cfg:
            merged_cfg["application"] = {}
        if "version" not in merged_cfg["application"]:
            merged_cfg["application"]["version"] = self.BROWSER_VERSION

        # Save merged config back to the instance so callers/other methods can access the full merged config
        self.config = merged_cfg

        # Save merged config
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(merged_cfg, f, indent=2)
