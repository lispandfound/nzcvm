# Root conftest that puts the repo root on sys.path so that
# `from tests.conftest import ...` resolves even though `tests/` has no
# `__init__.py`.  No other collection-side configuration lives here.
