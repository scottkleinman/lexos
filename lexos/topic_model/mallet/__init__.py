"""__init__.py."""

import re
import shlex
from collections import Counter
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, Popen, check_output
from typing import List

from spacy.tokens import Token
from wasabi import Printer

from . import scale_model


class Mallet:
    """A wrapper for the MALLET command line tool."""

    def __init__(self, model_dir: str, mallet_path: str = "mallet"):
        """Initialize the MALLET object.

        Args:
            model_dir (str): The directory to store the model.
            mallet_path (str): The path to the MALLET binary.
        """
        self.model_dir = model_dir
        self.mallet_path = mallet_path

    def import_data(self,
        docs: List[object],
        allowed: List[str] = None,
        remove_stops: bool = True,
        remove_punct: bool = True,
        use_lemmas: bool = False,
        **kwargs):
        """Import data into MALLET.

        Args:
            docs (List[object]): A list of spaCy documents.
            allowed (List[str]): A list of POS tags that are allowed.
            remove_stops (bool): Whether to remove stop words.
            remove_punct (bool): Whether to remove punctuation.
            use_lemmas (bool): Whether to replace tokens with lemmas.

        Notes:
            Creates a file containing one doc per line with each doc
            consisting of space-separated terms repeated however many
            times they occurred in the source doc. This file is then
            over-written by the MALLET import-file command, potentially
            using any MALLET command flags that are passed in (although
            most of the work is done by the first step in the process).
        """
        msg = Printer()
        if not Path(f"{self.model_dir}/data_skip.txt").is_file():
            msg.text("Bagifying data...")
            # Set the allowable tokens
            if allowed:
                is_allowed_getter = lambda token: token.pos_ in allowed
                Token.set_extension("is_allowed", getter=is_allowed_getter, force=True)
            else:
                Token.set_extension("is_allowed", default=True, force=True)
            bags = []
            # Get the token text for each doc
            for doc in docs:
                if use_lemmas:
                    tokens = [
                        token.lemma_ for token in doc
                        if token._.is_allowed
                        and token.is_stop != remove_stops
                        and token.is_punct != remove_punct
                    ]
                else:
                    tokens = [
                        token.text for token in doc
                        if token._.is_allowed
                        and token.is_stop != remove_stops
                        and token.is_punct != remove_punct
                    ]
                # Get the token counts
                counts = dict(Counter(tokens))
                # Create a bag with copies of each token occurring multiple times
                bag = []
                for k, v in counts.items():
                    repeated = f"{k} " * v
                    bag.append(repeated.strip())
                bags.append(" ".join(bag))
            # Write the data file with a bag for each document
            self.data_file = f"{self.model_dir}/data.txt"
            with open(self.data_file, "w", encoding="utf-8") as f:
                f.write("\n".join(bags))
        else:
            self.data_file = f"{self.model_dir}/data.txt"
        self.mallet_file = f"{self.model_dir}/import.mallet"
        # Build the MALLET import command
        opts = {
            "keep-sequence": True,
            "preserve-case": True,
            "remove-stopwords": False,
            "extra-stopwords": False,
            "token-regex": '"\S+"',
            "stoplist-file": None,
            }
        opts.update(kwargs)
        cmd_opts = []
        for k, v in opts.items():
            if v is not None:
                if v == True:
                    cmd_opts.append(f"--{k}")
                elif isinstance(v, str):
                    cmd_opts.append(f"--{k} {v}")
        mallet_cmd = f"{self.mallet_path}/mallet import-file --input {self.data_file} --output {self.mallet_file} "
        mallet_cmd += " ".join(cmd_opts)
        msg.text(f"Running {mallet_cmd}")
        mallet_cmd = shlex.split(mallet_cmd)
        # Perform the import
        try:
            # shell=True required to handle backslashes in token-regex
            output = check_output(mallet_cmd, stderr=STDOUT, shell=True, universal_newlines=True)
            msg.good("Import complete.")
        except CalledProcessError as e:
            output = e.output#.decode()
            msg.fail(output)

    def train(self,
                mallet_file: str = None,
                num_topics: int = 20,
                num_iterations: int = 1000,
                optimize_interval: int = 10,
                random_seed: int = None,
                **kwargs):
        """Train a model.

        Args:
            num_topics (int): The number of topics to train.
            num_iterations (int): The number of iterations to train.
            optimize_interval (int): The number of iterations between optimization.
            random_seed (int): The random seed to use.
        """
        msg = Printer()
        # Set the options
        try:
            if not mallet_file:
                mallet_file = self.mallet_file
        except AttributeError:
            msg.fail("Please supply an `input` argument with the path to your MALLET import file.")
        opts = {
            "input": mallet_file,
            "num-topics": str(num_topics),
            "num-iterations": str(num_iterations),
            "optimize-interval": str(optimize_interval),
            "random-seed": random_seed,
            "output-state": f"{self.model_dir}/state.gz",
            "output-topic-keys": f"{self.model_dir}/keys.txt",
            "output-doc-topics": f"{self.model_dir}/composition.txt",
            "word-topic-counts-file": f"{self.model_dir}/counts.txt",
            "output-topic-docs": f"{self.model_dir}/topic-docs.txt",
            "diagnostics-file": f"{self.model_dir}/diagnostics.xml"
        }
        opts.update(kwargs)
        cmd_opts = []
        for k, v in opts.items():
            if v is not None:
                if k == "random-seed":
                    v = str(v)
                if v == True:
                    cmd_opts.append(f"--{k}")
                elif isinstance(v, str):
                    cmd_opts.append(f"--{k} {v}")
        cmd_opts = " ".join(cmd_opts)
        mallet_cmd = f"{self.mallet_path}/mallet train-topics {cmd_opts}"
        msg.text(f"Running {mallet_cmd}\n")
        p = Popen(mallet_cmd, stdout=PIPE, stderr=STDOUT, shell=True)
        ll = []
        prog = re.compile(u'\<([^\)]+)\>')
        while p.poll() is None:
            l = p.stdout.readline().decode()
            print(l, end='')
            # Keep track of LL/topic.
            try:
                this_ll = float(re.findall('([-+]\d+\.\d+)', l)[0])
                ll.append(this_ll)
            except IndexError:  # Not every line will match.
                pass
            # Keep track of modeling progress
            try:
                this_iter = float(prog.match(l).groups()[0])
                progress = int(100. * this_iter/num_iterations)
                if progress % 10 == 0:
                    print('Modeling progress: {0}%.\r'.format(progress)),
            except AttributeError:  # Not every line will match.
                pass

    def scale(self, model_state_file: str = None, output_file: str = None):
        """Scale a model.

        Args:
            model_state_file (str): The path to a state_file.
            output_file (str): The path to an output file.
        """
        msg = Printer()
        msg.text("Processing...")
        if not model_state_file:
            model_state_file = f"{self.model_dir}/state.gz"
        if not output_file:
            output_file = f"{self.model_dir}/topic_scaled.csv"
        # try:
        # Convert the mallet output_state file to a pyLDAvis data object
        converted_data = scale_model.convert_mallet_data(model_state_file)
        # Get the topic coordinates in a dataframe
        topic_coordinates = scale_model.get_topic_coordinates(**converted_data)
        # Save the topic coordinates to a CSV file
        topic_coordinates.to_csv(output_file, index=False, header=False)
        msg.good("Done!")
        # except Exception:
        #     msg.fail("Failed!")
