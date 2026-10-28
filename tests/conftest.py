import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    # get_settings() calls load_dotenv(override=True), which would read the real
    # project .env and permanently mutate os.environ. Unit tests must be hermetic,
    # so neutralize it unless running integration tests (RUN_INTEGRATION=1).
    if os.environ.get("RUN_INTEGRATION") != "1":
        monkeypatch.setattr("finassist.config.load_dotenv", lambda *a, **k: False)
