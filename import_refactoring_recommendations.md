# Import Refactoring Recommendations

I have researched the Lexos codebase to identify the primary classes and functions for each subpackage. Based on this, I have developed a plan to implement top-level imports and ensure each subpackage provides a consistent public API.

The goal is to allow users to use Lexos with cleaner imports, such as:

```python
from lexos import Corpus, Scrubber, Tokenizer, DTM
```

instead of deeply nested imports like `from lexos.corpus.corpus import Corpus`.

## Plan: Top-level API Exposure and Documentation Alignment

Implement top-level imports in `lexos/__init__.py` and refine subpackage __init__.py files to expose core functionality, ensuring consistency across the library and compatibility with documentation tools.

**Steps**

### Phase 1: Core Subpackage Refinement

1. **lexos.corpus**: Update __init__.py to consistently expose `Corpus`, `Record`, `CorpusStats`, and `RecordsDict`.
2. **lexos.cutter**: Update __init__.py to export `TextCutter` and `TokenCutter`.
3. **lexos.visualization**: Update __init__.py to export `WordCloud` (from cloud.py) and `BubbleViz` (from bubbleviz.py).
4. **lexos.topwords**: Update __init__.py to export `TopWords` (base), `KeyTerms` (from keyterms.py), and statistical tests.
5. **lexos.rolling_windows**: Ensure `Windows` is correctly exported in __init__.py.

#### Extra Recommendations for Phase 1

1. `lexos.filter`
Currently has a docstring and `__all__`, but needs alignment with the new header style.
*   **Exports**: `BaseFilter`, `IsWordFilter`, `IsRomanFilter`, `IsStopwordFilter`.
*   **Action**: Update __init__.py to include the standardized "Phase 1 export surface" header.

2. `lexos.io`
Has a placeholder docstring (`"""__init__.py."""`) and `__all__`.
*   **Exports**: `BaseLoader`, `DataLoader`, `Loader`, `ParallelLoader`.
*   **Action**: Update __init__.py with a descriptive header and explicit export checklist.

3. `lexos.milestones`
Currently an empty shell (`"""__init__.py."""`). This package contains several useful milestone classes spread across multiple files.
*   **Proposed Exports**: `LineMilestones`, `SentenceMilestones`, `TokenMilestones`, `StringMilestones`.
*   **Action**: Initialize __init__.py to expose these classes at the top level.

4. `lexos.topic_modeling`
Currently an empty shell (`"""__init__.py."""`).
*   **Proposed Exports**: `Mallet` (from the `mallet` subpackage).
*   **Action**: Initialize __init__.py to expose `Mallet`.

### Phase 2: Top-Level API Implementation

1. Modify __init__.py to import and re-export the most frequently used classes:

- **Data Management**: `Corpus`, `Record`
- **Processing**: `Scrubber`, `Tokenizer`, `TextCutter`, `TokenCutter`
- **Analysis**: `DTM`, `Kwic`, `Windows`, `Mallet`
- **Visualization**: `WordCloud`, `BubbleViz`

### Phase 3: Documentation and Cleanup

1. **Circular Import Check**: Verify that top-level imports do not trigger circular dependencies, especially between `lexos.corpus` and `lexos.dtm`.
2. **Docstring Consistency**: Ensure all __init__.py files have updated docstrings explaining the public API at that level.
3. **User Guide Consistency**: Ensure all User Guide pages reflect the changes.
4. **Tutorial Consistency**: Ensure all Tutorial notebooks reflect the changes.

**Relevant files**

- __init__.py — Main entry point for top-level exports.
- __init__.py — Expose `Corpus` and `Record`.
- __init__.py — Expose `TextCutter` and `TokenCutter`.
- __init__.py — Expose `WordCloud` and `BubbleViz`.
- __init__.py — Expose `KeyTerms`.

**Verification**

1. **Import Tests**: Create a small script to verify that all intended classes are accessible via `from lexos import ...`.
2. **Existing Tests**: Run `pytest tests/` to ensure no regressions in existing code that might rely on specific import paths.
3. **Docstring Check**: Manually verify that `__all__` is defined in all modified __init__.py files to support `from lexos.subpackage import *`.

**Decisions**

- **Excluded**: Internal utility functions (e.g., `_create_dendrogram_traces`) will not be exposed at the top level to keep the API clean.
- **Naming**: `WordCloud` will be used instead of just `Cloud` in the visualization package for clarity.

**Further Considerations**

1. Would you like certain analysis tools (like specific statistical tests from `topwords`) to be available at the very top level, or keep those inside `lexos.topwords`?
2. Should we implement a "short-form" aliases (e.g., `from lexos import BC` for `BootstrapConsensus`) or stick to the full class names? My recommendation is to stick to full, descriptive class names.
