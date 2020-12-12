import sys
import numpy as np
import math
import copy
import random
import time

import os
import signal
from datetime import datetime
import subprocess
import uuid

import json
from pydoc import locate
import shutil


from es_maml import plain_maml
import es_maml




if __name__ == '__main__':

    # read in config
    with open('config.json') as f:
        config = json.load(f)

    # call plain_maml.run_maml(config)
    print("Starting run!")
    plain_maml.run_maml(config)
    print("DONE")













