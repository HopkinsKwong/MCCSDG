# ----> pytorch imports
import torch

# ----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import *
from utils.core_utils import _get_splits, _init_loss_function,_init_model,_init_optim,_init_loaders,_get_lr_scheduler
from utils.core_utils import _extract_survival_metadata,_summary
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args

import warnings

warnings.filterwarnings('ignore')


def _val_step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader,checkpoint_path):
    r"""
    Trains the model for the set number of epochs and validates it.

    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler
        - train_loader
        - val_loader

    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader)

    # for epoch in range(args.max_epochs):
    #     _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
    #     # _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
    #     # print('Val loss:', total_loss, ', val_c_index:', val_cindex)
    # # save the trained model
    # # 加载模型的状态字典
    # checkpoint_path = "/data/Survpath_com/results/results_titan_blca/tcga_blca__nll_surv_a0.5_lr5e-04_l2Weight_0.001_5foldcv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_100_fusion_None_modality_survpath_pathT_combine/s_0_checkpoint.pt"
    state_dict = torch.load(checkpoint_path)

    # 将状态字典加载到模型中
    model.load_state_dict(state_dict)

    results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory,
                                                                                                model, args.modality,
                                                                                                val_loader, loss_fn,
                                                                                                all_survival)

    print('Final Val c-index: {:.4f}'.format(val_cindex))
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex,
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))

    return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _val(datasets, cur, args,checkpoint_path):
    """
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int
        - args : argspace.Namespace

    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    # ----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)

    # ----> init loss function
    loss_fn = _init_loss_function(args)

    # ----> init model
    model = _init_model(args)

    # ---> init optimizer
    optimizer = _init_optim(args, model)

    # ---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    # ---> do train val
    results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _val_step(cur, args, loss_fn, model,
                                                                                           optimizer, lr_scheduler,
                                                                                           train_loader, val_loader,checkpoint_path=checkpoint_path)

    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)


def main(args,checkpoint_dir,ck_num:int):
    # ----> prep for 5 fold cv study
    # folds = _get_start_end(args)
    folds = [0]

    # ----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []
    checkpoints = []

    for i in folds:
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )

        print("Created train and val datasets for fold {}".format(i))

        for j in range(ck_num):
            ck_path = os.path.join(checkpoint_dir, f"s_{j}_checkpoint.pt")

            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _val(datasets, i, args,checkpoint_path=ck_path)

            all_val_cindex.append(val_cindex)
            all_val_cindex_ipcw.append(val_cindex_ipcw)
            all_val_BS.append(val_BS)
            all_val_IBS.append(val_IBS)
            all_val_iauc.append(val_iauc)
            all_val_loss.append(total_loss)
            checkpoints.append(f"s{j}")

            # write results to pkl
            filename = os.path.join(args.results_dir, 'checkpoint_s{}_results.pkl'.format(j))
            print("Saving results...")
            _save_pkl(filename, results)

    final_df = pd.DataFrame({
        'checkpoints': checkpoints,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        'val_BS': all_val_BS,
    })


    save_name = f'summary.csv'

    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    checkpoint_dir= "/data/Survpath_com/results/results_titan_blca/tcga_blca__nll_surv_a0.5_lr5e-04_l2Weight_0.001_traincv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_100_fusion_None_modality_survpath_pathT_combine/"
    ck_num= 5

    start = timer()

    # ----> read the args
    args = _process_args()

    # ----> Prep
    args = _prepare_for_experiment(args)

    # ----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat=True if "coattn" in args.modality else False,
        is_survpath=True if args.modality == "survpath" else False,
        type_of_pathway=args.type_of_path)

    # ---> perform the experiment
    results = main(args,checkpoint_dir=checkpoint_dir,ck_num=ck_num)

    # ---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))