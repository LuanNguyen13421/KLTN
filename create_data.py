import argparse
from utils.generate_dataset import Generate_Dataset


parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('--input', '--split', type=str, help ="Input video directory or file")
parser.add_argument('--output', type=str, default='dataset/data.h5', help="Output dataset file")
parser.add_argument('--extract-method', type=str, required=True, choices=['his', 'cnn'], help="Extract feature method (Color histogram / CNN)")
parser.add_argument('--bin', type=int, default=256, help="RGB BIN (If extract feature method is color histogram)")

args = parser.parse_args()
if __name__ == "__main__":
    gen = Generate_Dataset(args.input, args.output, args.extract_method, args.bin)
    gen.generate_dataset()
    gen.h5_file.close()