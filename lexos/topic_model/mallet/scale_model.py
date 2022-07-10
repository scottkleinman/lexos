"""scale_model.py."""
from typing import Callable

import gzip
import logging

import numpy as np
import pandas as pd
import sklearn.preprocessing

# Set fallback for MDS scaling
try:
    from sklearn.manifold import MDS, TSNE

    sklearn_present = True
except ImportError:
    sklearn_present = False
# from past.builtins import basestring
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


def __num_dist_rows__(array, ndigits: int = 2):
    """Check that all rows in a matrix sum to 1."""
    return array.shape[0] - int((pd.DataFrame(array).sum(axis=1) < 0.999).sum())


class ValidationError(ValueError):
    """Handle validation errors."""

    pass


def _input_check(
    topic_term_dists: pd.DataFrame,
    doc_topic_dists: pd.DataFrame,
    doc_lengths: list,
    vocab: list,
    term_frequency: int,
) -> list:
    """Check input for scale_model.

    Args:
        topic_term_dists (pd.DataFrame): Matrix of topic-term probabilities.
        doc_topic_dists (pd.DataFrame): Matrix of document-topic probabilities.
        doc_lengths (list): List of document lengths.
        vocab (list): List of vocabulary.
        term_frequency (int): Minimum number of times a term must appear in a document.

    Returns:
        list: List of errors.
    """
    ttds = topic_term_dists.shape
    dtds = doc_topic_dists.shape
    errors = []

    def err(msg):
        """Append error message."""
        errors.append(msg)

    if dtds[1] != ttds[0]:
        err(
            "Number of rows of topic_term_dists does not match number of columns of doc_topic_dists; both should be equal to the number of topics in the model."
        )

    if len(doc_lengths) != dtds[0]:
        err(
            "Length of doc_lengths not equal to the number of rows in doc_topic_dists; both should be equal to the number of documents in the data."
        )

    W = len(vocab)
    if ttds[1] != W:
        err(
            "Number of terms in vocabulary does not match the number of columns of topic_term_dists (where each row of topic_term_dists is a probability distribution of terms for a given topic)."
        )
    if len(term_frequency) != W:
        err(
            "Length of term_frequency not equal to the number of terms in the vocabulary (len of vocab)."
        )

    if __num_dist_rows__(topic_term_dists) != ttds[0]:
        err("Not all rows (distributions) in topic_term_dists sum to 1.")

    if __num_dist_rows__(doc_topic_dists) != dtds[0]:
        err("Not all rows (distributions) in doc_topic_dists sum to 1.")

    if len(errors) > 0:
        return errors


def _input_validate(*args) -> None:
    """Check input for scale_model."""
    res = _input_check(*args)
    if res:
        raise ValidationError("\n" + "\n".join([" * " + s for s in res]))


def _jensen_shannon(_P: np.array, _Q: np.array) -> float:
    """Calculate Jensen-Shannon Divergence.

    Args:
        _P (np.array): Probability distribution.
        _Q (np.array): Probability distribution.

    Returns:
        float: Jensen-Shannon Divergence.
    """
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def _pcoa(pair_dists: np.array, n_components: int = 2) -> np.array:
    """Perform Principal Coordinate Analysis.

    AKA Classical Multidimensional Scaling
    Code referenced from skbio.stats.ordination.pcoa
    https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py

    Args:
        pair_dists (np.array): Pairwise distances.
        n_components (int): Number of dimensions to reduce to.

    Returns:
        np.array: PCoA matrix.
    """
    # pairwise distance matrix is assumed symmetric
    pair_dists = np.asarray(pair_dists, np.float64)

    # perform SVD on double centred distance matrix
    n = pair_dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(pair_dists ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eig(B)

    # Take first n_components of eigenvalues and eigenvectors
    # sorted in decreasing order
    ix = eigvals.argsort()[::-1][:n_components]
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    # replace any remaining negative eigenvalues and associated eigenvectors with zeroes
    # at least 1 eigenvalue must be zero
    eigvals[np.isclose(eigvals, 0)] = 0
    if np.any(eigvals < 0):
        ix_neg = eigvals < 0
        eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
        eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

    return np.sqrt(eigvals) * eigvecs


def js_PCoA(distributions: np.array) -> np.array:
    """Perform dimension reduction.

    Works via Jensen-Shannon Divergence & Principal Coordinate Analysis
    (aka Classical Multidimensional Scaling)

    Args:
        distributions: (array-like, shape (`n_dists`, `k`)): Matrix of distributions probabilities.

    Returns:
        pcoa (np.array): (array, shape (`n_dists`, 2))

    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    return _pcoa(dist_matrix)


def js_MMDS(distributions: np.array, **kwargs) -> np.array:
    """Perform dimension reduction.

    Works via Jensen-Shannon Divergence & Metric Multidimensional Scaling

    Args:
        distributions (np.array): Matrix of distributions probabilities (array-like, shape (`n_dists`, `k`)).
        **kwargs (dict): Keyword argument to be passed to `sklearn.manifold.MDS()`

    Returns:
        mmds (np.array): (array, shape (`n_dists`, 2))

    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    model = MDS(n_components=2, random_state=0, dissimilarity="precomputed", **kwargs)
    return model.fit_transform(dist_matrix)


def js_TSNE(distributions, **kwargs) -> np.array:
    """Perform dimension reduction.

    Works via Jensen-Shannon Divergence & t-distributed Stochastic Neighbor Embedding

    Args:
        distributions (np.array): Matrix of distributions probabilities  (array-like, shape (`n_dists`, `k`)).
        **kwargs (dict): Keyword argument to be passed to `sklearn.manifold.MDS()`

    Returns:
        tsne (np.array): (array, shape (`n_dists`, 2))
    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    model = TSNE(n_components=2, random_state=0, metric="precomputed", **kwargs)
    return model.fit_transform(dist_matrix)


def _df_with_names(data, index_name: str, columns_name: str) -> pd.DataFrame:
    """Get a dataframe with names.

    Args:
        data (pd.DataFrame): Dataframe.
        index_name (str): Name of index.
        columns_name (str): Name of columns.

    Returns:
        pd.DataFrame: Dataframe with names.
    """
    if isinstance(data, pd.DataFrame):
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name) -> pd.Series:
    """Get a series with name.

    Args:
        data (pd.Series): Series.
        name (str): Name of series.

    Returns:
        pd.Series: Series with name.
    """
    if isinstance(data, pd.Series):
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def _topic_coordinates(
    mds: np.array, topic_term_dists: np.array, topic_proportion: np.array
) -> pd.DataFrame:
    """Get coordinates for topics.

    Args:
        mds (array, shape (`n_dists`, 2)): MDS coordinates.
        topic_term_dists (array, shape (`n_topics`, `n_terms`)): Topic-term distributions.
        topic_proportion (array, shape (`n_topics`)): Topic proportions.

    Returns:
        pd.DataFrame: Topic coordinates.
    """
    K = topic_term_dists.shape[0]
    mds_res = mds(topic_term_dists)
    assert mds_res.shape == (K, 2)
    mds_df = pd.DataFrame(
        {
            "x": mds_res[:, 0],
            "y": mds_res[:, 1],
            "topics": range(1, K + 1),
            "cluster": 1,
            "Freq": topic_proportion * 100,
        }
    )
    # note: cluster (should?) be deprecated soon. See: https://github.com/cpsievert/LDAvis/issues/26
    return mds_df


def get_topic_coordinates(
    topic_term_dists: np.array,
    doc_topic_dists: np.array,
    doc_lengths: list,
    vocab: list,
    term_frequency: list,
    mds: Callable = js_PCoA,
    sort_topics: bool = True,
) -> pd.DataFrame:
    """Transform the topic model distributions and related corpus.

    Creates the data structures needed for topic bubbles.

    Args:
        topic_term_dists (array-like, shape (`n_topics`, `n_terms`)): Matrix of topic-term probabilities where
            `n_terms` is `len(vocab)`.
        doc_topic_dists (array-like, shape (`n_docs`, `n_topics`)): Matrix of document-topic probabilities.
        doc_lengths : (array-like, shape `n_docs`): The length of each document, i.e. the number of words
            in each document. The order of the numbers should be consistent with the ordering of the docs in
            `doc_topic_dists`.
        vocab (array-like, shape `n_terms`): List of all the words in the corpus used to train the model.
        term_frequency (array-like, shape `n_terms`): The count of each particular term over the entire corpus.
            The ordering of these counts should correspond with `vocab` and `topic_term_dists`.
        mds (Callable): A function that takes `topic_term_dists` as an input and outputs a `n_topics` by `2`
            distance matrix. The output approximates the distance between topics. See `js_PCoA()` for details
            on the default function. A string representation currently accepts `pcoa` (or upper case variant),
            `mmds` (or upper case variant) and `tsne` (or upper case variant), if `sklearn` package is installed
            for the latter two.
        sort_topics (bool): Whether to sort topics by topic proportion (percentage of tokens covered). Set to
            `False` to to keep original topic order.

    Returns:
        scaled_coordinates (pd.DataFrame): A pandas dataframe containing scaled x and y coordinates.
    """
    # parse mds
    # if isinstance(mds, basestring):
    if isinstance(mds, (str,bytes)):
        mds = mds.lower()
        if mds == "pcoa":
            mds = js_PCoA
        elif mds in ("mmds", "tsne"):
            if sklearn_present:
                mds_opts = {"mmds": js_MMDS, "tsne": js_TSNE}
                mds = mds_opts[mds]
            else:
                logging.warning("sklearn not present, switch to PCoA")
                mds = js_PCoA
        else:
            logging.warning("Unknown mds `%s`, switch to PCoA" % mds)
            mds = js_PCoA

    topic_term_dists = _df_with_names(topic_term_dists, "topic", "term")
    doc_topic_dists = _df_with_names(doc_topic_dists, "doc", "topic")
    term_frequency = _series_with_name(term_frequency, "term_frequency")
    doc_lengths = _series_with_name(doc_lengths, "doc_length")
    vocab = _series_with_name(vocab, "vocab")
    _input_validate(
        topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
    )

    topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
    if sort_topics:
        topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
    else:
        topic_proportion = topic_freq / topic_freq.sum()

    topic_order = topic_proportion.index
    topic_term_dists = topic_term_dists.iloc[topic_order]

    scaled_coordinates = _topic_coordinates(mds, topic_term_dists, topic_proportion)

    return scaled_coordinates


def extract_params(statefile: str) -> tuple:
    """Extract the alpha and beta values from the statefile.

    Args:
        statefile (str): Path to statefile produced by MALLET.

    Returns:
        tuple: A tuple of (alpha (list), beta)
    """
    with gzip.open(statefile, "r") as state:
        params = [x.decode("utf8").strip() for x in state.readlines()[1:3]]
    return (list(params[0].split(":")[1].split(" ")), float(params[1].split(":")[1]))


def state_to_df(statefile: str) -> pd.DataFrame:
    """Transform state file into pandas dataframe.

    The MALLET statefile is tab-separated, and the first two rows contain the alpha and beta hypterparamters.

    Args:
        statefile (str): Path to statefile produced by MALLET.

    Returns:
        pd.DataFrame: The topic assignment for each token in each document of the model.
    """
    return pd.read_csv(statefile, compression="gzip", sep=" ", skiprows=[1, 2])


def pivot_and_smooth(
    df: pd.DataFrame,
    smooth_value: float,
    rows_variable: str,
    cols_variable: str,
    values_variable: str,
) -> pd.DataFrame:
    """Turn the pandas dataframe into a data matrix.

    Args:
        df (pd.DataFrame): The aggregated dataframe.
        smooth_value (float): Value to add to the matrix to account for the priors.
        rows_variable (str): The name of the dataframe column to use as the rows in the matrix.
        cols_variable (str): The name of the dataframe column to use as the columns in the matrix.
        values_variable (str): The name of the dataframe column to use as the values in the matrix.

    Returns:
        pd.DataFrame: A pandas matrix that has been normalized on the rows.
    """
    matrix = df.pivot(
        index=rows_variable, columns=cols_variable, values=values_variable
    ).fillna(value=0)
    matrix = matrix.values + smooth_value

    normed = sklearn.preprocessing.normalize(matrix, norm="l1", axis=1)

    return pd.DataFrame(normed)


def convert_mallet_data(state_file: str) -> dict:
    """Convert Mallet data to a structure compatible with pyLDAvis.

    Args:
        state_file (string): Mallet state file

    Returns:
        data (dict): A dict containing pandas dataframes for the pyLDAvis prepare method.
    """
    params = extract_params(state_file)
    alpha = [float(x) for x in params[0][1:]]
    beta = params[1]
    df = state_to_df(state_file)
    # Ensure that NaN is a string
    df["type"] = df.type.astype(str)
    # Get document lengths from statefile
    docs = df.groupby("#doc")["type"].count().reset_index(name="doc_length")
    # Get vocab and term frequencies from statefile
    vocab = df["type"].value_counts().reset_index()
    vocab.columns = ["type", "term_freq"]
    vocab = vocab.sort_values(by="type", ascending=True)
    phi_df = (
        df.groupby(["topic", "type"])["type"].count().reset_index(name="token_count")
    )
    phi_df = phi_df.sort_values(by="type", ascending=True)
    phi = pivot_and_smooth(phi_df, beta, "topic", "type", "token_count")
    theta_df = (
        df.groupby(["#doc", "topic"])["topic"].count().reset_index(name="topic_count")
    )
    theta = pivot_and_smooth(theta_df, alpha, "#doc", "topic", "topic_count")
    data = {
        "topic_term_dists": phi,
        "doc_topic_dists": theta,
        "doc_lengths": list(docs["doc_length"]),
        "vocab": list(vocab["type"]),
        "term_frequency": list(vocab["term_freq"]),
    }
    return data
