"""compare.py.

Last Updated: November 14, 2025
Last Tested: November 14, 2025
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, validate_call
from spacy.tokens import Doc

from lexos.exceptions import LexosException
from lexos.topwords import TopWords
from lexos.util import ensure_list


class Compare(BaseModel):
    """Compare class for calculating token statistics in documents against a corpus or other classes of documents."""

    calculator: TopWords
    data: list = Field(default_factory=list)
    results: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _calculate(
        self, target_docs: list[Doc], comparison_docs: list[Doc]
    ) -> list[tuple]:
        """Calculate statistics using the provided calculator.

        Args:
            target_docs (list[Doc]): Target documents
            comparison_docs (list[Doc]): Comparison documents

        Returns:
            list[tuple]: List of (term, score) tuples
        """
        # Create a new instance of the calculator with the specific docs
        calculator_class = type(self.calculator)

        # Get the configuration from the original calculator
        config = {}
        for field in self.calculator.model_fields:
            if field not in ["target_docs", "comparison_docs", "topwords"]:
                config[field] = getattr(self.calculator, field)

        # Create new instance with target and comparison docs
        new_calculator = calculator_class(
            target_docs=target_docs, comparison_docs=comparison_docs, **config
        )

        return new_calculator.topwords

    def _format_output(
        self,
        results: dict,
        output_format: str,
        label_key: str = "doc_label",
    ) -> dict | pd.DataFrame | list[dict]:
        """Format comparison results into the requested output format.

        Args:
            results (dict): Dictionary of results with structure:
                {label: {"comparison_class": str, "topwords": list[tuple]}} or
                {label: list[tuple]} for simple format
            output_format (str): Desired output format ('dict', 'dataframe', 'list_of_dicts')
            label_key (str): Key name for the label column ('doc_label' or 'class_label')

        Returns:
            dict | pd.DataFrame | list[dict]: Formatted results
        """
        if output_format == "dict":
            return results
        elif output_format == "dataframe":
            return self._to_dataframe(results, label_key)
        elif output_format == "list_of_dicts":
            return self._to_list_of_dicts(results, label_key)
        else:
            raise LexosException(
                f"Unsupported output_format: {output_format}. Supported formats are 'dict', 'dataframe', and 'list_of_dicts'."
            )

    def _to_dataframe(
        self, results: dict, label_key: str = "doc_label"
    ) -> pd.DataFrame:
        """Convert results dictionary to DataFrame.

        Args:
            results (dict): Results dictionary
            label_key (str): Key name for the label column

        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        rows = []
        for label, result in results.items():
            # Handle structured results with comparison_class
            if isinstance(result, dict) and "topwords" in result:
                comparison_class = result["comparison_class"]
                topwords = result["topwords"]
                for term, score in topwords:
                    rows.append(
                        {
                            label_key: label,
                            "comparison_class": comparison_class,
                            "term": term,
                            "score": score,
                        }
                    )
            # Handle simple results (list of tuples)
            else:
                row = {label_key: label}
                for term, score in result:
                    row[term] = score
                rows.append(row)

        df = pd.DataFrame(rows)
        # For simple format, set label as index and fill 0.0
        if label_key in df.columns and "comparison_class" not in df.columns:
            df = df.set_index(label_key)
            df = df.fillna(0.0)
        return df

    def _to_list_of_dicts(
        self, results: dict, label_key: str = "doc_label"
    ) -> list[dict]:
        """Convert results dictionary to list of dictionaries.

        Args:
            results (dict): Results dictionary
            label_key (str): Key name for the label column

        Returns:
            list[dict]: List of dictionaries with term statistics
        """
        list_of_dicts = []
        for label, result in results.items():
            # Handle structured results with comparison_class
            if isinstance(result, dict) and "topwords" in result:
                comparison_class = result["comparison_class"]
                topwords = result["topwords"]
                for term, score in topwords:
                    list_of_dicts.append(
                        {
                            label_key: label,
                            "comparison_class": comparison_class,
                            "term": term,
                            "score": score,
                        }
                    )
            # Handle simple results (list of tuples)
            else:
                for term, score in result:
                    list_of_dicts.append(
                        {label_key: label, "term": term, "score": score}
                    )
        return list_of_dicts

    def convert_output(
        self,
        output_format: str,
        label_key: str = None,
    ) -> dict | pd.DataFrame | list[dict]:
        """Convert cached results to a different output format without recalculating.

        Args:
            output_format (str): Desired output format ('dict', 'dataframe', 'list_of_dicts')
            label_key (str, optional): Key name for the label column ('doc_label' or 'class_label').
                If not provided, will auto-detect based on results structure.

        Returns:
            dict | pd.DataFrame | list[dict]: Results in the requested format

        Raises:
            LexosException: If no results are cached (must run a comparison method first)
        """
        if not self.results:
            raise LexosException(
                "No results cached. Run a comparison method first (document_to_corpus, "
                "documents_to_classes, or classes_to_classes)."
            )

        # Auto-detect label_key if not provided
        if label_key is None:
            # Check if data contains class_label to determine the type
            if self.data and "class_label" in self.data[0]:
                # For documents_to_classes and classes_to_classes, check first result
                first_result = next(iter(self.results.values()))
                if isinstance(first_result, dict) and "topwords" in first_result:
                    # This could be either - check if we have unique classes
                    unique_classes = {item["class_label"] for item in self.data}
                    if len(self.results) == len(unique_classes):
                        label_key = "class_label"  # classes_to_classes
                    else:
                        label_key = "doc_label"  # documents_to_classes
                else:
                    label_key = "doc_label"  # document_to_corpus
            else:
                label_key = "doc_label"

        return self._format_output(self.results, output_format, label_key)

    def _create_data_dict(
        self,
        docs: list[str | Doc | dict[str, str]],
        doc_labels: list[str] = None,
        doc_classes: list[str] = None,
    ) -> list[dict]:
        """Build a database from documents, their labels, and their classes.

        Args:
            docs (list[str | Doc | dict[str, str]]): A list of documents, which
                can be strings, spaCy Doc objects, or dicts containing 'doc', 'doc_label', and 'class_label'. If doc_label is missing in the dicts, it will be auto-generated.
            doc_labels (list[str], optional): A list of document labels corresponding to
                each document. If not provided, doc_labels will be auto-generated.
            doc_classes (list[str], optional): A list of class labels with indexes
                corresponding to the order of docs. If docs is a list of strings or spaCy Docs, and doc_classes is not provided, an error will be raised.

        Returns:
            list[dict[str, str | Doc]]: A list of dicts like with "doc_label", "class_label", and "doc".
        """
        # Return dicts directly if they have appropriate labels
        if isinstance(docs[0], dict):
            return self._validate_docs_dict(docs)

        # Process lists, ensuring that doc_labels and doc_classes are correct
        elif isinstance(docs, list):
            return self._validate_docs_list(docs, doc_labels, doc_classes)

        else:
            raise LexosException(
                "Docs must be a list of dicts, strings, or Doc objects."
            )

    def _validate_docs_dict(self, docs: list[dict]) -> list[dict[str, str | Doc]]:
        """Validate the structure of a docs dictionary.

        Args:
            docs (list[dict]): A list of dictionaries to validate.

        Returns:
            list[dict]: The validated list of dictionaries.
        """
        required_keys = {"class_label", "doc"}
        for i, doc in enumerate(docs):
            if not required_keys.issubset(doc.keys()):
                raise LexosException(
                    f"When passing docs as a list of dicts, the dicts must contain the keys: {', '.join(required_keys)}"
                )
            # If doc_label is missing, add incremental "Doc {i}"
            if "doc_label" not in doc:
                docs[i]["doc_label"] = f"Doc {i + 1}"
        return docs

    def _validate_docs_list(
        self,
        docs: list[dict],
        doc_labels: list[str] = None,
        doc_classes: list[str] = None,
    ) -> list[dict[str, str | Doc]]:
        """Validate the structure of a docs dictionary.

        Args:
            docs (list[dict]): A list of dictionaries to validate.
            doc_labels (list[str], optional): A list of document labels
                corresponding to each document. If not provided, doc_labels will be auto-generated.
            doc_classes (list[str], optional): A list of class labels with indexes
                corresponding to the order of docs. If  doc_classes is not provided, an error will be raised.

        Returns:
            list[dict]: The validated list of dictionaries.
        """
        data = []
        for i, doc in enumerate(docs):
            doc_data = {}
            # Get the doc_labels if provided in a list of strings, else generate defaults
            if not doc_labels:
                doc_data["doc_label"] = f"Doc {i}"
            else:
                doc_data["doc_label"] = doc_labels[i]
            # If class_labels are provided
            if doc_classes:
                # If the doc is a Doc and the extension is set, get the extension value
                try:
                    if isinstance(doc, Doc) and Doc.has_extension(doc_classes[i]):
                        doc_class = doc._.get(doc_classes[i])
                    # Otherwise, treat as literal strings
                    else:
                        doc_class = doc_classes[i]
                except IndexError:
                    raise LexosException(
                        "The docs and doc_classes values must be the same length if using a list of literal strings for classes."
                    )
                doc_data["class_label"] = doc_class
            else:
                raise LexosException(
                    "Document classes labels must be provided using `doc_classes`."
                )
            doc_data["doc"] = doc
            data.append(doc_data)
        return data

    @validate_call(config=model_config)
    def document_to_corpus(
        self,
        corpus: str | Doc | list[str | Doc],
        doc_labels: list[str] = None,
        output_format: str = "dict",
    ) -> dict[str, list[tuple]] | pd.DataFrame | list[dict]:
        """Calculate statistics for tokens in each document against the rest of the corpus.

        Each document is compared individually to all other documents in the corpus combined.

        Args:
            corpus (str | Doc | list[str | Doc] | list[dict]): A single document or a list of documents to compare.
            doc_labels (list[str], optional): A list of document labels corresponding to
                each document. If not provided, labels will be auto-generated as "Doc 1", "Doc 2", etc.
            output_format (str, optional): The desired output format. Options are:
                - 'dict': Dictionary mapping doc labels to lists of (term, score) tuples
                - 'dataframe': pandas DataFrame with terms as columns and docs as rows
                - 'list_of_dicts': List of dicts with 'doc_label', 'term', and 'score' keys
                Defaults to 'dict'.

        Returns:
            dict[str, list[tuple]] | pd.DataFrame | list[dict]:
                - If 'dict': {doc_label: [(term, score), ...]}
                - If 'dataframe': DataFrame with doc_label index and term columns
                - If 'list_of_dicts': [{'doc_label': str, 'term': str, 'score': float}, ...]
        """
        corpus = ensure_list(corpus)
        if len(corpus) < 2:
            raise LexosException(
                "Corpus must contain at least two documents for comparison."
            )

        # Generate labels if not provided
        if doc_labels is None:
            doc_labels = [f"Doc {i + 1}" for i in range(len(corpus))]

        # Assign the doc_labels and docs to the data attribute so it can be accessed
        self.data = [
            {"doc_label": label, "doc": doc} for label, doc in zip(doc_labels, corpus)
        ]

        # Calculate the results
        results = {}
        for i, doc in enumerate(corpus):
            target_docs = [doc]
            comparison_docs = corpus[:i] + corpus[i + 1 :]
            scores = self._calculate(target_docs, comparison_docs)
            results[doc_labels[i]] = scores

        # Cache the raw results (sorted by doc labels)
        self.results = dict(sorted(results.items()))
        return self._format_output(self.results, output_format, label_key="doc_label")

    @validate_call(config=model_config)
    def documents_to_classes(
        self,
        docs: str | Doc | list[str | Doc] | list[dict],
        doc_labels: list[str] | dict[str, str] = None,  # Doc names
        class_labels: list[str] = None,  # Class names
        output_format: str = "dict",
    ) -> dict[str, dict] | pd.DataFrame | list[dict]:
        """Calculate statistics for each document against all documents in other classes.

        Each document is compared to documents in all classes except its own class.

        Args:
            docs (str | Doc | list[str | Doc] | list[dict]): A single document or a list of documents to compare.
            doc_labels (list[str] | dict[str, str], optional): A list of document labels corresponding to each document,
                or a dict mapping document identifiers to labels. If not provided, labels will be auto-generated.
            class_labels (list[str], optional): A list of class labels with
                indexes corresponding to the order of docs. If not provided, an error will be raised.
            output_format (str, optional): The desired output format. Options are:
                - 'dict': Nested dict with comparison_class and topwords
                - 'dataframe': pandas DataFrame with doc_label, comparison_class, term, and score columns
                - 'list_of_dicts': List of dicts with 'doc_label', 'comparison_class', 'term', and 'score' keys
                Defaults to 'dict'.

        Returns:
            dict[str, dict] | pd.DataFrame | list[dict]:
                - If 'dict': {doc_label: {'comparison_class': str, 'topwords': [(term, score), ...]}}
                - If 'dataframe': DataFrame with columns [doc_label, comparison_class, term, score]
                - If 'list_of_dicts': [{'doc_label': str, 'comparison_class': str, 'term': str, 'score': float}, ...]
        """
        # Ensure docs is a list
        docs = ensure_list(docs)

        # Validate inputs
        if len(docs) < 2:
            raise LexosException("At least two documents are required for comparison.")

        # Create the data dict from the inputs
        self.data = self._create_data_dict(docs, doc_labels, class_labels)

        # Group documents by their class {"class_label": [docs]}
        docs_by_class = {}
        for item in self.data:
            docs_by_class.setdefault(item["class_label"], []).append(item["doc"])

        # Validate that we have at least 2 classes
        if len(docs_by_class) < 2:
            raise LexosException(
                f"At least two different classes are required for comparison. Found {len(docs_by_class)} class(es): {list(docs_by_class.keys())}"
            )

        # Pre-compute comparison docs for each class (cache to avoid rebuilding)
        comparison_cache = {
            cls: [
                doc
                for other_cls, docs in docs_by_class.items()
                if other_cls != cls
                for doc in docs
            ]
            for cls in docs_by_class
        }

        # Compute results
        results = {}
        for item in self.data:
            doc = item["doc"]
            doc_label = item["doc_label"]
            doc_class = item["class_label"]

            # Get the comparison classes (all classes except the document's own class)
            comparison_classes = [
                cls for cls in docs_by_class.keys() if cls != doc_class
            ]

            # Get pre-computed comparison docs for this class
            comparison_docs = comparison_cache[doc_class]

            # Calculate scores for the current document
            scores = self._calculate([doc], comparison_docs)

            # Store results with comparison class information
            results[doc_label] = {
                "comparison_class": ", ".join(comparison_classes),
                "topwords": scores,
            }

        # Cache the raw results
        self.results = results
        return self._format_output(self.results, output_format, label_key="doc_label")

    @validate_call(config=model_config)
    def classes_to_classes(
        self,
        docs: str | Doc | list[str | Doc] | list[dict],
        doc_labels: list[str] | dict[str, str] = None,  # Doc names
        class_labels: list[str] = None,  # Class names
        output_format: str = "dict",
    ) -> dict[str, dict] | pd.DataFrame | list[dict]:
        """Calculate statistics comparing each class (all its documents combined) to all documents in other classes.

        All documents within each class are treated as a single unit and compared to all documents
        in all other classes combined.

        Args:
            docs (str | Doc | list[str | Doc] | list[dict]): A single document or a list of documents to compare.
            doc_labels (list[str] | dict[str, str], optional): A list of document labels corresponding to each document,
                or a dict mapping document identifiers to labels. If not provided, labels will be auto-generated.
            class_labels (list[str], optional): A list of class labels with
                indexes corresponding to the order of docs. If not provided, an error will be raised.
            output_format (str, optional): The desired output format. Options are:
                - 'dict': Nested dict with comparison_class and topwords
                - 'dataframe': pandas DataFrame with class_label, comparison_class, term, and score columns
                - 'list_of_dicts': List of dicts with 'class_label', 'comparison_class', 'term', and 'score' keys
                Defaults to 'dict'.

        Returns:
            dict[str, dict] | pd.DataFrame | list[dict]:
                - If 'dict': {class_label: {'comparison_class': str, 'topwords': [(term, score), ...]}}
                - If 'dataframe': DataFrame with columns [class_label, comparison_class, term, score]
                - If 'list_of_dicts': [{'class_label': str, 'comparison_class': str, 'term': str, 'score': float}, ...]
        """
        # Ensure docs is a list
        docs = ensure_list(docs)

        # Validate inputs
        if len(docs) < 2:
            raise LexosException("At least two documents are required for comparison.")

        # Create the data dict from the inputs
        self.data = self._create_data_dict(docs, doc_labels, class_labels)

        # Group documents by their class {"class_label": [docs]}
        docs_by_class = {}
        for item in self.data:
            docs_by_class.setdefault(item["class_label"], []).append(item["doc"])

        # Validate that we have at least 2 classes
        if len(docs_by_class) < 2:
            raise LexosException(
                f"At least two different classes are required for comparison. Found {len(docs_by_class)} class(es): {list(docs_by_class.keys())}"
            )

        # Pre-compute comparison docs for each class (cache to avoid rebuilding)
        comparison_cache = {
            cls: [
                doc
                for other_cls, docs in docs_by_class.items()
                if other_cls != cls
                for doc in docs
            ]
            for cls in docs_by_class
        }

        # Compute results - compare all docs in each class to all docs in other classes
        results = {}
        for class_label, class_docs in docs_by_class.items():
            # Get the comparison classes (all classes except the current class)
            comparison_classes = [
                cls for cls in docs_by_class.keys() if cls != class_label
            ]

            # Get pre-computed comparison docs for this class
            comparison_docs = comparison_cache[class_label]

            # Calculate scores for all documents in this class combined
            scores = self._calculate(class_docs, comparison_docs)

            # Store results with comparison class information
            results[class_label] = {
                "comparison_class": ", ".join(comparison_classes),
                "topwords": scores,
            }

        # Cache the raw results
        self.results = results
        return self._format_output(self.results, output_format, label_key="class_label")

    @validate_call(config=model_config)
    def get_class(self, doc_label: str) -> str:
        """Get the class label for a given document label.

        Args:
            doc_label (str): The label of the document.

        Returns:
            str: The class label of the document.
        """
        for item in self.data:
            if item["doc_label"] == doc_label:
                return item["class_label"]
        raise LexosException(f"Document with label '{doc_label}' not found in data.")

    @validate_call(config=model_config)
    def get_docs_by_class(self, class_label: str = None) -> dict[str, list[Doc]]:
        """Get documents grouped by their class labels.

        Args:
            class_label (str, optional): If provided, only returns documents for this class.

        Returns:
            dict[str, list[Doc]]: A dictionary mapping class labels to lists of documents.
        """
        docs_by_class = {}
        for item in self.data:
            docs_by_class.setdefault(item["class_label"], []).append(item["doc_label"])
        if class_label is not None:
            return {class_label: docs_by_class.get(class_label, [])}
        return docs_by_class
