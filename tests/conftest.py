"""Keep the host singleton and test databases out of the source checkout."""
import os
from tempfile import TemporaryDirectory

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    directory = TemporaryDirectory(prefix="mr-memory-tests-")
    original = os.environ.get("ASTRBOT_ROOT")
    os.environ["ASTRBOT_ROOT"] = directory.name

    def restore():
        if original is None:
            os.environ.pop("ASTRBOT_ROOT", None)
        else:
            os.environ["ASTRBOT_ROOT"] = original
        directory.cleanup()

    config.add_cleanup(restore)
