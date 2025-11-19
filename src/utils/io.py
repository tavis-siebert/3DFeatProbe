# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

def safe_makedirs(path):
    if not path:
        logging.warning("safe_makedirs called with an empty path. No operation performed.")
        return False

    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logging.error(f"Failed to create directory '{path}'. Reason: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected error occurred while creating directory '{path}'. Reason: {e}")
        raise