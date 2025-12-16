"""Pytest configuration for taubench tests."""

import pytest


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API calls",
    )


@pytest.fixture
def run_api_tests(request):
    """Check if API tests should run."""
    return request.config.getoption("--run-api-tests")


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for anyio tests."""
    return "asyncio"
