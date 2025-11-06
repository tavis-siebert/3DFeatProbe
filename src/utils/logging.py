# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import atexit
import functools
from .io import safe_makedirs

# cache the opened file object, so that different calls
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = open(filename, mode="a", buffering=1024)
    atexit.register(io.close)
    return io

def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
    all_ranks: bool = False,
):
    """
    Setup console + optional file logging without external dependencies.
    """
    log_filename = None
    if output_dir:
        safe_makedirs(output_dir)
        if rank == 0:
            log_filename = os.path.join(output_dir, "log.txt")
        elif all_ranks:
            log_filename = os.path.join(output_dir, f"log_{rank}.txt")

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    fmt = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(fmt)

    # clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logging.root.handlers = []
    logger.root.handlers = []

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(
        log_level_primary if rank == 0 else log_level_secondary
    )
    logger.addHandler(console_handler)

    # file handler (optional)
    if log_filename is not None:
        file_stream = _cached_log_stream(log_filename)
        file_handler = logging.StreamHandler(file_stream)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level_primary)
        logger.addHandler(file_handler)

    # Make this logger the root logger
    logging.root = logger
    return logger
