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

        self.set_dataset_dir(self.reg_factor, self.video_type)

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

def str2bool(v):
    """ 
    Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(parse=True, **optional_kwargs):
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
    parser.add_argument('--split-id', type = int, default = 0, help = "Split index [0-4] (default: 0)")
    parser.add_argument('-m', '--metric', type = str, required = True, choices = ['tvsum', 'summe'], help = "Evaluation metric ['tvsum', 'summe']")
    
    # Mode
    parser.add_argument('--mode', type = str, default = 'train', help ='Mode for the configuration [train | test]')
    parser.add_argument('--verbose', type = str2bool, default = 'true', help = 'Print or not training messages')
    parser.add_argument('--video_type', type = str, default = 'SumMe', help = 'Dataset to be used')

    # Model
    parser.add_argument('--input_size', type = int, default = 1024, help = 'Feature size expected in the input')
    parser.add_argument('--block_size', type = int, default = 60, help = "Size of blocks used inside the attention matrix")
    parser.add_argument('--init_type', type = str, default = "xavier", help = 'Weight initialization method')
    parser.add_argument('--init_gain', type = float, default = 1.4142, help = 'Scaling factor for the initialization methods')

    # Train
    parser.add_argument('--n_epochs', type = int, default = 400, help = 'Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 20, help = 'Size of each batch in training')
    parser.add_argument('--seed', type = int, default = 12345, help = 'Chosen seed for generating random numbers')
    parser.add_argument('--clip', type = float, default = 5.0, help = 'Max norm of the gradients')
    parser.add_argument('--lr', type = float, default = 5e-4, help = 'Learning rate used for the modules')
    parser.add_argument('--l2_req', type = float, default = 1e-5, help = 'Weight regularization factor')
    parser.add_argument('--reg_factor', type = float, default = 0.6, help = 'Length regularization factor')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)