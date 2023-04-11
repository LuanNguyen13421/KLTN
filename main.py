from utils import Logger, read_json, write_json, save_checkpoint
from rewards import compute_reward
from configs import get_config
import summarizer_module
import vsum_tools

if __name__ == '__main__':
    config = get_config(mode='train')
    print(config)
    num = int(input())