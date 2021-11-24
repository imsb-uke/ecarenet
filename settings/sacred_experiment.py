'''----------------------------------------------------------------------------
IMPORTS
-----------------------------------------------------------------------------'''

from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sacred import Experiment
import os

from settings.default_settings import (EXPERIMENTS_DIR, DEFAULT_CONFIG_FILE)

# define sacred expirment (ex) and add data_ingredientdocker
ex = Experiment('ecarenet', interactive=True, base_dir=os.path.abspath(os.path.join(os.getcwd(), '..')))
ex.add_config(DEFAULT_CONFIG_FILE)


if not os.path.isdir(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)
ex.observers.append(FileStorageObserver.create(EXPERIMENTS_DIR))
# if ex.configurations[0]._conf['platform']['use_mongo']:
#    ex.observers.append(MongoObserver(url=platform["db_url"]))

ex.capture_out_filter = apply_backspaces_and_linefeeds



