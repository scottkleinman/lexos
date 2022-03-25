"""__init__.py."""

import re
import shlex
import shutil
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output

TEMPLATE_DIR = Path(__file__).parent / "template"
# TEMPLATE_PATH = ROOT / "dfrb_template.zip"

class DfrBrowser:
    """DfrBrowser class."""

    def __init__(self,
                 model_dir: str = ".",
                 model_state_file: str = "state.gz",
                 model_scaled_file: str = "topic_scaled.csv",
                 template_dir: str = TEMPLATE_DIR) -> None:
        """Initialize DfrBrowser object."""
        self.template_dir = template_dir
        self.model_dir = model_dir
        self.model_state_file = f"{model_dir}/{model_state_file}"
        self.model_scaled_file = f"{model_dir}/{model_scaled_file}"
        self.browser_dir = f"{model_dir}/dfr_browser"
        self.data_dir = f"{self.browser_dir}/data"
        self.num_topics = None # How to get this?

        # Make a browser directory and copy the template into it
        if not Path(self.browser_dir).exists():
            self._copy_template()

        # Create dfr-browser files using python script
        self._prepare_data()

        # Copy scaled file into data dir
        shutil.copy(self.model_scaled_file, self.data_dir)

        # Move meta.csv to data_dir, zip up, and rename, delete meta.csv copy
        self._move_metadata()

        # Update assets
        self._update_assets()


    def _copy_template(self):
        """Copy the template directory to the browser directory."""
        shutil.copytree(Path(self.template_dir), Path(self.browser_dir))

    def _prepare_data(self):
        """Prepare the data for the dfr-browser visualization."""
        Path(f"{self.data_dir}").mkdir(parents=True, exist_ok=True)
        prepare_data_script = f"python {self.browser_dir}/bin/prepare-data"
        cmd = " ".join([
            prepare_data_script,
            "convert-state",
            self.model_state_file,
            "--tw", f"{self.data_dir}/tw.json",
            "--dt", f"{self.data_dir}/dt.json.zip"
        ])
        cmd = shlex.split(cmd)
        try:
            output = check_output(cmd, stderr=STDOUT, shell=True, universal_newlines=True)
        except CalledProcessError as e:
            output = e.output
        print(output)
        cmd = " ".join([prepare_data_script, "info-stub", "-o", f"{self.data_dir}/info.json"])
        cmd = shlex.split(cmd)
        try:
            output = check_output(cmd, stderr=STDOUT, shell=True, universal_newlines=True)
        except CalledProcessError as e:
            output = e.output
        print(output)

    def _move_metadata(self):
        """Move meta.csv to data_dir, zip up, rename, and delete meta.csv copy."""
        meta_zip = f"{self.data_dir}/meta.csv.zip"
        if Path(meta_zip).exists():
            Path(meta_zip).unlink()
        browser_meta_file = f"{self.model_dir}/meta.csv"
        shutil.copy(browser_meta_file, self.data_dir)
        try:
            shutil.make_archive(f"{self.data_dir}/meta.csv", "zip", self.data_dir, "meta.csv")
        except OSError as err:
            print("Error writing meta.csv.zip")
            print(err)

    def _update_assets(self):
        """Update browser assets."""
        # Tweak default index.html to link to JSON, not JSTOR
        with open(f"{self.browser_dir}/index.html", "r") as f:
            filedata = f.read().replace("on JSTOR", "JSON")
        with open(f"{self.browser_dir}/index.html", "w") as f:
            f.write(filedata)
        # Tweak js file to link to the domain
        with open(f"{self.browser_dir}/js/dfb.min.js.custom", "r", encoding="utf-8") as f:
            filedata = f.read()
        pat = r't\.select\(\"#doc_remark a\.url\"\).attr\(\"href\", .+?\);'
        new_pat = r'var doc_url = document.URL.split("modules")[0] + "project_data"; t.select("#doc_remark a.url")'
        new_pat += r'.attr("href", doc_url + "/" + e.url);'
        filedata = re.sub(pat, new_pat, filedata)
        with open(f"{self.browser_dir}/js/dfb.min.js", "w", encoding="utf-8") as f:
            f.write(filedata)

    def run(self, port: int = 8080) -> None:
        """Run the dfr-browser.

        This might work on the Jupyter port, but it might not.
        """
        # run_server = f"python {self.browser_dir}/bin/server"
        import os
        import sys
        import threading
        import time
        import webbrowser as w
        from http.server import HTTPServer, SimpleHTTPRequestHandler

        # set up the HTTP server and start it in a separate daemon thread
        httpd = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True

        # if startup time is too long we might want to be able to quit the program
        current_dir = os.getcwd()
        try:
            os.chdir(self.browser_dir)
            thread.start()
        except KeyboardInterrupt:
            httpd.shutdown()
            os.chdir(current_dir)
            sys.exit(0)

        # wait until the webserver finished starting up (maybe wait longer or shorter...)
        time.sleep(3)

        # start sending requests
        w.open(f"http://127.0.0.1:{port}/")

