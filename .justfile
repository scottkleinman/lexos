# List all recipes
@_:
    just --list

# Run pytest with optional arguments
test *args:
    uv run pytest {{ args }}

# Run pytest with coverage
cov *args :
    uv run pytest {{ args }} --cov --cov-report=term-missing

# Bump dependency versions
upgrade:
    uv sync --upgrade
