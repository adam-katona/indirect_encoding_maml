

from es_maml.omniglot import omniglotNShot
import sys


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage:  python download_omiglot.py \"path/to/store/dataset\"")
        sys.exit()

    path = sys.argv[1]
    

    # we instanciate the class, so it will:
    #   download the raw data
    #   transform the data to 28x28
    #   save the whole dataset as a numpy array into path/omniglot.npy

    # We call this so it will save the numpy array,
    # we won't use the class further,
    # the only important parametes are root and imgsz
    dataset = omniglotNShot.OmniglotNShot(root=path, batchsz=1, n_way=1, k_shot=1, k_query=1, imgsz=28)


