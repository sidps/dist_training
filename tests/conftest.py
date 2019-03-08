import logging
import torch
from main import parse_args
from pytest import fixture
from src.iris.data import Data
from src.iris.model import Architecture


@fixture(scope='class')
def mock_args(tmpdir_factory):
    p = tmpdir_factory.mktemp('test_log_path')
    return parse_args(argv=['--log-path', p.strpath])


@fixture(scope='class', autouse=True)
def setup_seed(mock_args):
    torch.manual_seed(mock_args.seed)
    return


@fixture(scope='class')
def sample_data(mock_args):
    return Data(mock_args)


@fixture(scope='class')
def sample_architecture():
    return Architecture


@fixture(scope='session')
def test_logger():
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    return logger
