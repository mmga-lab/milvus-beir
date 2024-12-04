import pytest
import random
import string
from pymilvus import MilvusClient

# Define the default value
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_TOKEN = "root:Milvus"

def pytest_addoption(parser):
    parser.addoption(
        "--milvus-uri",
        action="store",
        default=DEFAULT_MILVUS_URI,
        help="URI for connecting to Milvus server",
    )
    parser.addoption(
        "--milvus-token",
        action="store",
        default=DEFAULT_MILVUS_URI,
        help="URI for connecting to Milvus server",
    )


@pytest.fixture
def milvus_uri(request):
    return request.config.getoption("--milvus-uri")

@pytest.fixture
def milvus_token(request):
    return request.config.getoption("--milvus-token")

@pytest.fixture
def collection_name():
    # Generate a random 8-character string
    chars = string.ascii_letters + string.digits
    random_str = "".join(random.choice(chars) for _ in range(8))
    return f"test_collection_{random_str}"


@pytest.fixture
def milvus_client(milvus_uri, milvus_token):
    client = MilvusClient(uri=milvus_uri, token=milvus_token)
    return client


@pytest.fixture
def test_corpus():
    return {
        "doc1": {"title": "Test Document 1", "text": "This is a test document one"},
        "doc2": {"title": "Test Document 2", "text": "This is a test document two"},
        "doc3": {"title": "Test Document 3", "text": "This is a test document three"},
    }


@pytest.fixture
def test_queries():
    return {"q1": "test query one", "q2": "test query two"}
