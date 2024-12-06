import numpy as np
from transformer_block import *
from utils import *
from networks import *
from train_model import *
from cross_validation import *
from prepare_data_DEAP import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-montage', type=int, default=3, choices=[1, 2, 3], help="Input 1 or 2 or 3")
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V'])
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=3)
    parser.add_argument('--emb-size', type=int, default=64)
    parser.add_argument('--GCN-layer-num', type=int, default=2)
    parser.add_argument('--segment-length', type=int, default=2)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--resolution' , type=float, default=1)
    parser.add_argument('--min-size' , type=float, default=2)
    parser.add_argument('--n-subgraph', type=int, default=4)
    parser.add_argument('--dropout-MHAttention', type=float, default=0.1)
    parser.add_argument('--dropout-FC', type=float, default=0.3)
    parser.add_argument('--dropout-LPEEGNet', type=float, default=0.3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--random-seed', type=int, default=2023)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--patient', type=int, default=20)
    parser.add_argument('--max-epoch', type=int, default=2)
    parser.add_argument('--n-splits1', type=int, default=2)
    parser.add_argument('--optimizerM', type=int, default=2)
    parser.add_argument('--weight-decay', type=float, default=5E-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--max-epoch-cmb', type=int, default=4)
    parser.add_argument('--model', type=str, default='LPEEGNet')


    # parser.
    args = parser.parse_args()
    if args.input_montage == 1:
        ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                    'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                    'CP2', 'P4', 'P8', 'PO4', 'O2']
        parser.add_argument('--ch-names', type=list, default=ch_names)
        parser.add_argument('--trails', type=int, default=40)
        parser.add_argument('--sfreq', type=int, default=128)
        parser.add_argument('--dataset-name', type=str, default='DEAP')
        parser.add_argument('--channel-num', type=int, default=32)
        parser.add_argument('--num-class', type=int, default=2)
        parser.add_argument('--classes', type=list, default=[0,1])
        parser.add_argument('--subjects', type=int, default=32)
        parser.add_argument('--save-path', default='./save/DEAP/')
        parser.add_argument('--load-path', default='./save/DEAP/max-acc.pth')
        parser.add_argument('--load-ee-path', default='./save/DEAP/min-ee-loss1.pth')
        parser.add_argument('--load-path-final', default='./save/DEAP/final_model.pth')
        parser.add_argument('--load-ee-path-final', default='./save/DEAP/final_model_ee.pth')

    else:
        trails = 45 if args.input_montage == 2 else 72
        dataset_name = 'SEED' if args.input_montage == 2 else 'SEED-IV'
        numclass = 3 if args.input_montage == 2 else 4
        classes_ = [0,1,2] if args.input_montage == 2 else [0,1,2,3]
        path = './save/SEED/' if args.input_montage == 2 else './save/SEED-IV/'
        path1 = './save/SEED/max-acc.pth' if args.input_montage == 2 else './save/SEED-IV/max-acc.pth'
        path2 = './save/SEED/min-ee-loss1.pth' if args.input_montage == 2 else './save/SEED-IV/min-ee-loss1.pth'
        path3 = './save/SEED/final_model.pth' if args.input_montage == 2 else './save/SEED-IV/final_model.pth'
        path4 = './save/SEED/final_model_ee.pth' if args.input_montage == 2 else './save/SEED-IV/final_model_ee.pth'
        sfreq = 200
        ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
                    'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
                    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
                    'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
        parser.add_argument('--ch-names', type=list, default=ch_names)
        parser.add_argument('--trails', type=int, default=trails)
        parser.add_argument('--sfreq', type=int, default=sfreq)
        parser.add_argument('--dataset-name', type=str, default=dataset_name)
        parser.add_argument('--channel-num', type=int, default=62)
        parser.add_argument('--num-class', type=int, default=numclass)
        parser.add_argument('--classes', type=list, default=classes_)
        parser.add_argument('--subjects', type=int, default=15)
        parser.add_argument('--save-path', default=path)
        parser.add_argument('--load-path', default=path1)
        parser.add_argument('--load-ee-path', default=path2)
        parser.add_argument('--load-path-final', default=path3)
        parser.add_argument('--load-ee-path-final', default=path4)

    args = parser.parse_args()
    sub_to_run = np.arange(args.subjects)
    # pd = PrepareData(args)
    # pd.run(sub_to_run,split=True)

    save_path = os.getcwd()
    path_name = 'data_{}_{}'.format(args.dataset_name, 'extracted')
    save_path = os.path.join(save_path, path_name, 'other_parameters.mat')
    dataset = scio.loadmat(save_path)
    num_segment = int(dataset['num_segment'])
    Label = dataset['Label']
    parser.add_argument('--num-segment', type=int, default=num_segment)
    parser.add_argument('--Label', type=list, default=Label)
    args = parser.parse_args()

    # ConnectivityForTrail(args.dataset_name, sub_to_run, args.ch_names, args.trails, args.sfreq)
    # Bregion_dividion_CNM(args.dataset_name, sub_to_run, args.ch_names, args.resolution, args.min_size)
    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=10)


