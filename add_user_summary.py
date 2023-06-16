import h5py
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser("Code adding user summary to created dataset (Only for SumMe or TVSum dataset)")
parser.add_argument('--input', type=str, help="input dataset")
parser.add_argument('--output', type=str, default='', help="output dataset")
parser.add_argument('--dataset-name', type=str, required = True, choices = ['tvsum', 'summe'], help="Dataset's name: tvsum or summe")
args = parser.parse_args()

def readSUMME(file, fileoutput):
    dataset = h5py.File('user_summary_h5_file/summe_user_summary.h5', 'r')
    writeData = h5py.File(file, 'r')
    count = 0
    with h5py.File(fileoutput, "w") as w:
        for video in dataset.keys():
            for tempVideo in writeData.keys():
                if dataset[video]['video_name'][...] == writeData[tempVideo]['video_name'][...]:
                        data_of_name = dataset[video]['user_summary'][...]
                        w.create_dataset(tempVideo + '/user_summary', data=data_of_name)
                        w.create_dataset(tempVideo + '/picks', data=writeData[tempVideo]['picks'][...])
                        w.create_dataset(tempVideo + '/features', data=writeData[tempVideo]['features'][...])
                        w.create_dataset(tempVideo + '/change_points', data=writeData[tempVideo]['change_points'][...])
                        w.create_dataset(tempVideo + '/n_frame_per_seg', data=writeData[tempVideo]['n_frame_per_seg'][...])
                        w.create_dataset(tempVideo + '/n_frames', data=writeData[tempVideo]['n_frames'][...])
                        w.create_dataset(tempVideo + '/video_name', data=writeData[tempVideo]['video_name'][...])
                        w.create_dataset(tempVideo + '/fps', data=writeData[tempVideo]['fps'][...])
                        count = count + 1
    print('Number of videos add user summary: ' + str(count))
    dataset.close()
    writeData.close()
    os.remove(args.input)

def readTVSUM(file, fileoutput):
    dataset = h5py.File('user_summary_h5_file/tvsum_user_summary.h5', 'r')
    writeData = h5py.File(file, 'r')
    count = 0
    with h5py.File(fileoutput, "w") as w:
        for video in dataset.keys():
            for tempVideo in writeData.keys():
                if abs(int(dataset[video]['n_frames'][...]) - int(writeData[tempVideo]['n_frames'][...])) < 2:
                        data_of_name = dataset[video]['user_summary'][...]
                        data_of_name = data_of_name.astype(np.float32)
                        w.create_dataset(tempVideo + '/user_summary', data=data_of_name)
                        w.create_dataset(tempVideo + '/picks', data=writeData[tempVideo]['picks'][...])
                        w.create_dataset(tempVideo + '/features', data=writeData[tempVideo]['features'][...])
                        w.create_dataset(tempVideo + '/change_points', data=writeData[tempVideo]['change_points'][...])
                        w.create_dataset(tempVideo + '/n_frame_per_seg', data=writeData[tempVideo]['n_frame_per_seg'][...])
                        w.create_dataset(tempVideo + '/n_frames', data=writeData[tempVideo]['n_frames'][...])
                        w.create_dataset(tempVideo + '/video_name', data=writeData[tempVideo]['video_name'][...])
                        w.create_dataset(tempVideo + '/fps', data=writeData[tempVideo]['fps'][...])
                        count = count + 1
    print('Number of videos add user summary: ' + str(count))
    dataset.close()
    writeData.close()
    os.remove(args.input)

def main():
    if args.dataset_name == 'summe':
         readSUMME(args.input, args.output)
    elif args.dataset_name == 'tvsum':
         readTVSUM(args.input, args.ouput)
    else:
         print('Your dataset cannot add user summary!\n')
         

if __name__ == "__main__":
    main()