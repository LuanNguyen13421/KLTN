import argparse
import torch
from pathlib import Path
import pprint

save_dir = Path('./Summaries/')

class Config(object):
    def __init__(self, **kwargs):
        """ Configuration Class: set kwargs as class attributes with setattr. """
        self.score_dir, self.save_dir = None, None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type = 'SumMe'):
        """ 
        Function that sets as class attributes the necessary directories for logging important training information.

        :param float reg_factor: The utilized length regularization factor.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        """
        self.score_dir = save_dir.joinpath(video_type, 'results/split' + str(self.split_index))
        self.save_dir = save_dir.joinpath(video_type, 'models/split' + str(self.split_index))

    def __repr__(self):
        """ Pretty-print configurations in alphabetical order. """
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str

def get_config(**optional_kwargs):
    """ 
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with ATTENTION and REINFORCE")

    # Dataset options
    parser.add_argument('-d', '--dataset', type = str, required = True, help = "Path to h5 dataset (required)")
    parser.add_argument('-s', '--split', type = str, required = True, help = "Path to split file (required)")
    parser.add_argument('--split_id', type = int, default = 0, help = "Split index [0-4] (default: 0)")
    parser.add_argument('--video_type', type = str, required = True, choices = ['tvsum', 'summe'], help = "Dataset to be used ['tvsum', 'summe']")
    
    # Mode
    parser.add_argument('--mode', type = str, default = 'train', help ='Mode for the configuration [train | test]')
    parser.add_argument('--verbose', action = 'store_true', help = 'Print or not print the training messages')

    # Model
    parser.add_argument('--input_size', type = int, default = 1024, help = 'Feature size expected in the input')
    parser.add_argument('--block_size', type = int, default = 60, help = "Size of blocks used inside the attention matrix")

    # Optimization options
    parser.add_argument('--lr', type = float, default = 1e-05, help = "Learning rate (default: 1e-05)")
    parser.add_argument('--weight_decay', type = float, default = 1e-05, help = "Ưeight decay rate (default: 1e-05)")
    parser.add_argument('--max_epoch', type = int, default = 60, help = "Maximum epoch for training (default: 60)")
    parser.add_argument('--stepsize', type = int, default = 30, help = "How many steps to decay learning rate (default: 30)")
    parser.add_argument('--gamma', type = float, default = 0.1, help = "Learning rate decay (default: 0.1)")
    parser.add_argument('--num_episode', type = int, default = 5, help = "Number of episodes (default: 5)")
    parser.add_argument('--beta', type = float, default = 0.01, help = "Weight for summary length penalty term (default: 0.01)")

    # Misc
    parser.add_argument('--seed', type = int, default = 1, help = "Random seed (default: 1)")
    parser.add_argument('--gpu', type = str, default = '0', help = "Which gpu devices to use")
    parser.add_argument('--use_cpu', action = 'store_true', help = "Use cpu device")
    parser.add_argument('--evaluate', action = 'store_true', help = "Whether to do evaluation only")
    parser.add_argument('--resume', type = str, default = '', help = "Path to resume file")
    parser.add_argument('--save_results', action = 'store_true', help = "Whether to save output results")

    kwargs = parser.parse_args()
    
    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)