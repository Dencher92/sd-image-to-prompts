import json
import logging
import os
import tempfile
from logging.handlers import QueueHandler
from multiprocessing import Process, Queue, current_process
from typing import Any, Dict

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import log_artifact
from pythonjsonlogger import jsonlogger

# general logging functions
def configure_aggregator_logger(
    queue: Queue,
    log_dir_path: str = 'main.log',
    log_level: int = logging.DEBUG
):
    os.makedirs(log_dir_path, exist_ok=True)
    logger = logging.getLogger('ChildLogAggregator')
    datefmt='%Y-%m-%d %H:%M:%S'

    fileHandler = logging.FileHandler(os.path.join(log_dir_path, 'agg.log'))
    file_format_str = '%(message)%(levelname)%(name)%(asctime)'
    fileFormatter = jsonlogger.JsonFormatter(file_format_str, datefmt=datefmt)
    fileHandler.setFormatter(fileFormatter)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    stream_format_str = '[%(name)s %(levelname)s] [%(asctime)s] %(message)s'
    streamFormatter = logging.Formatter(stream_format_str, datefmt=datefmt)
    streamHandler.setFormatter(streamFormatter)
    logger.addHandler(streamHandler)

    logger.setLevel(log_level)
    logger.propagate = False
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)


def start_aggregator_logger(
    queue: Queue,
    log_dir_path: str = 'main.log',
    log_level: int = logging.DEBUG
):
    process = Process(
        target=configure_aggregator_logger,
        args=(queue, log_dir_path, log_level)
    )
    process.start()
    return process


def configure_child_logger(
    log_queue: Queue,
    log_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Configure a logger for a child process. The logger will push all messages to the queue, which will be handled by the
    aggregator logger. The aggregator logger will then write the messages to a file and/or stdout.
    Takes the root logger, adds a QueueHandler to it and renames it to the process name.
    """
    process = current_process()
    logger = logging.getLogger()
    if logger.name == process.name:
        # logger already configured
        return logger

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(log_level)
    logger.name = process.name
    # logger.propagate = False
    return logger


def configure_main_logger(
    log_queue: Queue,
    log_level: int = logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(log_level)
    logger.name = 'Main'
    logger.propagate = False
    return logger


# formatting functions
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def persist_object(obj: Any, path: str, as_yaml: bool = False, mode: str = 'w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    elif isinstance(obj, matplotlib.figure.Figure):
        obj.savefig(path)
    else:
        with open(path, mode) as f:
            if as_yaml:
                yaml.dump(obj, f, allow_unicode=True)
            else:
                json.dump(obj, f, cls=NpEncoder)


def flatten_dict(d: Dict, sep='.') -> Dict:
    df = pd.json_normalize(d, sep=sep)
    return df.to_dict(orient='records')[0]


# mlflow specific functions
def log_params_to_mlflow(
    config: Dict[str, Any],
) -> None:
    """Log parameters to MLFlow. Allows nested dictionaries."""
    flat_dict = flatten_dict(config)
    # mlflow can only process 100 parameters at once
    keys = sorted(flat_dict.keys())
    batch_size = 100
    for start in range(0, len(keys), batch_size):
        keys_batch = keys[start:start + batch_size]
        mlflow.log_params({k: flat_dict[k] for k in keys_batch})


def log_object_as_artifact(obj: Any, artifact_name: str, as_yaml: bool = False, mode='w'):
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = os.path.join(tmp_dir, artifact_name)
        persist_object(obj, artifact_path, as_yaml, mode)
        log_artifact(artifact_path)
