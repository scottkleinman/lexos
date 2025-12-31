# List all recipes
@_:
    just --list

# Run pytest with optional arguments
test *args:
    uv run pytest {{ args }}

# Run pytest with coverage
cov *args :
    uv run pytest {{ args }} --cov --cov-report=term-missing

# Run mkdocs serve
serve:
    uv run mkdocs serve -f doc_src/mkdocs.yml

# Bump dependency versions
upgrade:
    uv sync --upgrade
