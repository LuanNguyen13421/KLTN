import argparse
from create_data_utils.generate_dataset import Generate_Dataset


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('--input', '--split', type=str, help="input video")
parser.add_argument('--output', type=str, default='', help="out data")
parser.add_argument('--timeDir', type=str, default='', help="time processing dir")
parser.add_argument('--bin', type=int, default=256, help="RGB BIN")
parser.add_argument('--extract_method', type=str, required = True, choices = ['his', 'cnn'], help="extract feature method")

args = parser.parse_args()
if __name__ == "__main__":
    gen = Generate_Dataset(args.input, args.output, args.timeDir, args.extract_method, args.bin)
    gen.generate_dataset()
    gen.h5_file.close()