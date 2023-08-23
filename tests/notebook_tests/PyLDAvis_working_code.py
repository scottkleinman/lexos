# Extend the Model for PyLDAvis
import logging
import numpy as np

from gensim import utils
from gensim import corpora
from gensim.models import basemodel
from gensim.models.ldamodel import LdaModel
from gensim.utils import revdict

import pyLDAvis
import pyLDAvis.gensim_models

logger = logging.getLogger(__name__)


# 1. Get the data as a list of lists of lemmas, where each internal list is a doc
data = [[token.text for token in doc] for doc in docs]

# 2. Get id2word (a dictionary mapping word ids to words)
id2word = corpora.Dictionary(data)

# 3. Next, we apply the doc2bow function to convert the texts into the bag-of-words (BoW) format,
#    This is the corpus (Term Document Frequency) a list of (token_id, token_count) tuples.
corpus = [id2word.doc2bow(text) for text in data]

# 4. We need to get our MALLET model into Gensim format. We do this with the LdaModel class.
class LdaMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    """Python wrapper for MALLET models."""

    def __init__(
        self,
        mallet_path,
        mallet_state_file,
        corpus=None,
        num_topics=100,
        alpha=50,
        id2word=None,
        workers=4,
        optimize_interval=0,
        iterations=1000,
        topic_threshold=0.0,
        random_seed=0,
    ):
        """Initialize the model.

        Args:
            mallet_path (str): Path to the mallet binary, e.g. `/home/username/mallet-2.0.7/bin/mallet`.
            mallet_state_file (str): Path to the trained mallet state file.
            corpus (Optional[Iterable[int, int]]): Collection of texts in BoW format.
            num_topics (Optional[int]): Number of topics.
            alpha (Optional[int]): Alpha parameter of LDA.
            id2word (Optional[gensim.corpora.dictionary.Dictionary]): Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`.
            workers (Optional[int]): Number of threads that will be used for training.
            optimize_interval (Optional[int]):  Optimize hyperparameters every `optimize_interval` iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization).
            iterations (Optional[int]): Number of training iterations.
            topic_threshold (Optional[float]): Threshold of the probability above which we consider a topic.
            random_seed (Optional[int]): Random seed to ensure consistent results, if 0 - use system clock.
        """
        self.mallet_path = mallet_path
        self.mallet_state_file = mallet_state_file
        self.id2word = id2word
        if self.id2word is None:
            logger.warning(
                "No word id mapping provided; initializing from corpus, assuming identity."
            )
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())
        if self.num_terms == 0:
            raise ValueError("Cannot compute LDA over an empty collection (no terms).")
        self.num_topics = num_topics
        self.topic_threshold = topic_threshold
        self.alpha = alpha
        self.workers = workers
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        self.random_seed = random_seed
        self.word_topics = self.load_word_topics()
        # For backward compatibility
        self.wordtopics = self.word_topics

    def load_word_topics(self):
        """Load words X topics matrix from the mallet state  file.

        Returns:
            numpy.ndarray: Matrix words X topics.

        """
        logger.info("loading assigned topics from %s", self.mallet_state_file)
        word_topics = np.zeros((self.num_topics, self.num_terms), dtype=np.float64)
        if hasattr(self.id2word, "token2id"):
            word2id = self.id2word.token2id
        else:
            word2id = revdict(self.id2word)

        with utils.open(self.mallet_state_file, "rb") as fin:
            _ = next(fin)  # header
            self.alpha = np.fromiter(next(fin).split()[2:], dtype=float)
            assert (
                len(self.alpha) == self.num_topics
            ), "mismatch between MALLET vs. requested topics"
            _ = next(fin)  # noqa:F841 beta
            for lineno, line in enumerate(fin):
                line = utils.to_unicode(line)
                doc, source, pos, typeindex, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved LdaMallet class. Handles backwards compatibility from
        older LdaMallet versions which did not use random_seed parameter.
        """
        model = super(LdaMallet, cls).load(*args, **kwargs)
        if not hasattr(model, "random_seed"):
            model.random_seed = 0

        return model


# 5. We can now create the LDA model.
mallet_path = "C:/mallet/mallet-2.0.8/bin"
state_file = "topic_model/state.gz"
mallet_model = LdaMallet(mallet_path, state_file, num_topics=20, id2word=id2word)
# If we leave out id2word, the model will infer it from the corpus.

# 6. Now we convert the LdaMallet model to a Gensim LdaModel
model_gensim = LdaModel(
    id2word=mallet_model.id2word,
    num_topics=mallet_model.num_topics,
    alpha=mallet_model.alpha,
    eta=0,
    iterations=1000,
    gamma_threshold=0.001,
    dtype=np.float32,
)
model_gensim.sync_state()
model_gensim.state.sstats = mallet_model.word_topics

# 7. Create the pyLDAvis visualisation
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model_gensim, corpus, id2word)
# Additional flags that may be useful:
# sort_topics=False
# mds="tsne|mmds"
# vis.show(view=True)
# vis.topic_order
# vis.to_json()
# pyLDAvis.save_html(vis, "use_lda.html")
