import os

'''-----------------------------------------------------------------------------
DIRECTORIES AND FILES
-----------------------------------------------------------------------------'''
PROJECT_DIR      = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR      = os.path.join(PROJECT_DIR, '..')
print('project dir: ', PROJECT_DIR)
print(os.listdir(PROJECT_DIR))

EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, 'experiments')
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_DIR, 'config.yaml')
