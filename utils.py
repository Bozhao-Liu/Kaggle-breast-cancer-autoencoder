import json
import logging
import os
import shutil
import torch
import numpy as np
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, model_dir, network):
        json_path = os.path.join(model_dir, network)
        json_file = os.path.join(json_path, 'params.json')
        logging.info("Loading json file {}".format(json_file))
        assert os.path.isfile(json_file), "Can not find File {}".format(json_file)
        with open(json_file) as f:
            params = json.load(f)

            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
        
    
def set_logger(model_dir, network, level = 'info'):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    log_path = os.path.join(model_dir, network)
    assert os.path.isdir(log_path), "Can not find Path {}".format(log_path)
    log_path = os.path.join(log_path, 'train.log')
    print('Saving {} log to {}'.format(level, log_path))
    level = level.lower()
    logger = logging.getLogger()
    if level == 'warning':
        level = logging.WARNING
    elif level == 'debug':
        level = logging.DEBUG
    elif level == 'error':
        level = logging.ERROR
    elif level == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def set_params(model_dir, network):
	params = Params(model_dir, network)

	# use GPU if available
	params.cuda = torch.cuda.is_available()

	# Set the random seed for reproducible experiments
	torch.manual_seed(230)
	if params.cuda: 
		torch.cuda.manual_seed(230)

	return params

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

	
def get_checkpointname(args, n_lf, checkpoint_type, CViter):
	checkpointpath = os.path.join(args.model_dir, args.network)
	checkpointpath = os.path.join(checkpointpath, 'Checkpoints' + str(checkpoint_type))
	checkpointfile = os.path.join(checkpointpath, 
				'{network}_{LrD}_{cv_iter}_{n_lf}.pth.tar'.format(network = args.network, 
									     LrD = args.lrDecay,
									     cv_iter = '_'.join(tuple(map(str, CViter))), n_lf = n_lf))
	return checkpointpath, checkpointfile									     

