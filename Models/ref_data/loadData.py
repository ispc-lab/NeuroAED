from ref_data.__local__ import implemented_datasets
from ref_data.wholeframe import WholeFrame
from ref_data.allcubconvec import Allcubconvec
from ref_data.sigactcuboid import SigActCuboid


def load_dataset(dataset_name, dataset_dir, pretrain=False):

    assert dataset_name in implemented_datasets

    if dataset_name == "wholeframe":
        data_loader = WholeFrame()
    
    if dataset_name == "allcubconvec":
        data_loader = Allcubconvec()

    if dataset_name == "sigactcuboid":
        data_loader = SigActCuboid(dataset_dir)

    return data_loader

    # if dataset_name == "neighbour":
    #     data_loader = Neighbour_DataLoader

    # if dataset_name == "entropy":
    #     data_loader = Entropy_DataLoader

    # if dataset_name == "cubhisto":
    #     data_loader = CubHisto_DataLoader

    # if dataset_name == "continue20":
    #     data_loader = Continue20_DataLoader
    
    # if dataset_name == "allcubconvec":
    #     data_loader = AllCubConVec_DataLoader

    # load data with data loader
    # learner.load_data(data_loader=data_loader, pretrain=pretrain)
