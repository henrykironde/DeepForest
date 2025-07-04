# Fixtures model to only download model once
# download latest release
import pytest
from deepforest import main
from deepforest import utilities
from deepforest import get_data
from deepforest import _ROOT
import os
import urllib

collect_ignore = ['setup.py']


@pytest.fixture(scope="session")
def config():
    config = utilities.load_config()
    config.train.fast_dev_run = True
    config.batch_size = 1 #True Why is this true?
    return config


@pytest.fixture(scope="session")
def download_release():
    print("running fixtures")
    try:
        main.deepforest().load_model("weecology/deepforest-tree")
    except urllib.error.URLError:
        # Add a edge case in case no internet access.
        pass


@pytest.fixture(scope="session")
def ROOT():
    return _ROOT


@pytest.fixture(scope="session")
def two_class_m():
    m = main.deepforest(num_classes=2, label_dict={"Alive": 0, "Dead": 1})
    m.config.train.csv_file = get_data("testfile_multi.csv")
    m.config.train.root_dir = os.path.dirname(get_data("testfile_multi.csv"))
    m.config.train.fast_dev_run = True
    m.config.batch_size = 2
    m.config.validation.csv_file = get_data("testfile_multi.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("testfile_multi.csv"))
    m.config.validation.val_accuracy_interval = 1

    m.create_trainer()

    return m


@pytest.fixture(scope="session")
def m(download_release):
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.train.fast_dev_run = True
    m.config.batch_size = 2
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.workers = 0
    m.config.validation.val_accuracy_interval = 1
    m.config.train.epochs = 2

    m.create_trainer()
    m.load_model("weecology/deepforest-tree")

    return m
