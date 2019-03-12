import sys
import numpy as np 
import torch 
import torchvision
from IPython import embed
from utils import get_onehot4DNA
    
def data_preprocess(dir='../../Data/a_b_chain/'):
    with open(dir + 'SRR_Acc_List.txt') as f:
        SRR_list = f.readlines()
    raw_data_a = []
    raw_data_b = []
    for SRR_id in SRR_list:
        SRR_id = SRR_id.strip()
        with open(dir + SRR_id + '.fastq') as fi:
            fastq = fi.readlines()
            # print(SRR_id + ' ' + str(len(fastq)))
            for (i, line) in enumerate(fastq): 
                line = line.strip()
                if i % 8 == 1: 
                    a = line.strip('N')
                if i % 8 == 5:
                    b = line.strip('N')
                    raw_data_a.append(a)
                    raw_data_b.append(b)

    # padding method directly use 'N' to fulfill the rest part at tail
    count = len(raw_data_a)
    raw_data = np.concatenate([raw_data_a, raw_data_b])
    length = max([len(DNA) for DNA in raw_data])
    len_file = open("length.txt", "w")
    for DNA in raw_data:
        print(len(DNA), file=len_file, flush=True)
    print("pair:", count)
    print("DNA length: %d" % length)

    # FIXME: naively use left-ajustment, be careful about this 
    raw_data = np.array([DNA.ljust(length, 'N') for DNA in raw_data])
    data = get_onehot4DNA(raw_data)
    # FIXME: temprally ignore BLOSUM matrix
    # data = to_BLOSUM(data)

    # concatenate a, b chain
    # FIXME: smaller length for debugging
    length = 600
    data_a, data_b = data[:count,:length], data[count:,:length]
    data = np.concatenate((data_a, data_b),axis=1).reshape(count, 1, -1, 5)
    np.random.shuffle(data)

    # split the dataset into train, validateation and test 
    train_count = int(count * 0.6)
    validate_count = int(count * 0.3)
    train_data = data[:train_count]
    validate_data = data[train_count:train_count+validate_count]

    print("train data:", train_data.shape)
    print("train:", train_count)
    print("validate:", validate_count)
    return train_data, validate_data, length

if __name__ == "__main__":
    data_preprocess()
