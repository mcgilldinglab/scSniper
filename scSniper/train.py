from running_utils.training_loop import *
from running_utils.eval_loop import *
from running_utils.model import AE_Classifier

import click
import pandas as pd
from running_utils.misc import *
import os
import scanpy as sc
from dnnlib import EasyDict
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam

import torch.nn as nn
from focal_loss.focal_loss import FocalLoss
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
import pickle
@click.command()
# Model-related.
@click.option('--encoder_dict', help='Output dimension of encoder layers', metavar='DICT', type=parse_dict_of_int_lists,
              default="{RNA:[128,128,50],ADT:[64,64,50]}", show_default=True)
@click.option('--decoder_dict', help='Output dimension of decoder layers', metavar='DICT', type=parse_dict_of_int_lists,
              default="{RNA:[128,128],ADT:[64,64]}", show_default=True)
@click.option('--classifier_interlayers_dims', help='Output dimension of classifier layers', metavar='DICT',type=parse_dict_of_int_lists,
              default="{Classifier:[32,10]}", show_default=True)

# Data-related.
@click.option('--data', help='Path to the dataset (H5AD)', metavar='H5AD', type=str, required=True)
@click.option('--categorical_covariate', help='obs key to covariate such as batch', metavar='STR', type=str, default=None,)
@click.option('--modality_keys', help='h5ad obsm keys to each modality, if you want to use .X, please input "X".Example: {RNA:X,ADT:Antibody Capture}}', metavar='STR', type=parse_dict_of_strings, default=None,required=True)
@click.option('--num_class', help='Number of label class', metavar="INT", type=int,
              default=2, show_default=True)
@click.option('--class_label', help='obs key to class label', metavar='STR', type=str, default=None, required=True)
@click.option('--celltype', help='Cell type key in obs, if not provided, assume there is one cell type', metavar='STR', type=str, default=None)

# Training-related.
@click.option('--seed', help='Random seed', metavar="INT", type=int, default=None)
@click.option('--batch_size', help='Input batch size', metavar="INT", type=int, default=128)
@click.option('--device', help='Device to use (cuda or cpu)', metavar='cuda|cpu', type=str, default='cuda')
@click.option('--batch_size_fine_tune', help='Input batch size during cell type specific fine tune', metavar="INT", type=int, default=128)
@click.option("--learning_rate", help="Learning rate", metavar="FLOAT", type=float, default=1e-3)
@click.option("--learning_rate_fine_tune", help="Learning rate during cell type specific fine tune", metavar="FLOAT", type=float, default=1e-4)
@click.option("--iteraction_multiplier", help="Iteraction multiplier", metavar="INT", type=int, default=1000)
@click.option("--early_stop_tolerance_final_step", help="Early stop tolerance", metavar="INT", type=int, default=100)
@click.option("--warm_up", help="Number of warm up iteractions", metavar="FLOAT", type=int, default=1000)
@click.option("--iteraction_fine_tune", help="Number of iteractions during cell type specific fine tune", metavar="INT", type=int, default=10000)
@click.option("--lambd", help="Weight of reconstruction in fine-tuneing", metavar="FLOAT", type=float, default=0.1)


#IO-related
@click.option('--save_folder_dir', help='Path to save model', metavar='STR', type=str, default=None)
def main(**kwargs):
    opts = EasyDict(kwargs)
    seed = opts.seed
    if seed is not None:
        set_seed(seed)

    # Load dataset.
    train_data = sc.read_h5ad(opts.data)

    # Setting hyperparameters
    num_class = opts.num_class
    dtype = np.float32
    #num_batch = opts.batch
    batch_size = opts.batch_size
    if opts.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("cuda is not available")
    device = torch.device(opts.device)
    modality = list(opts.encoder_dict.keys())
    print("num_class", num_class)
    #print("num_batch", num_batch)

    # Check folder if exists or not
    if opts.save_folder_dir is None:
        save_folder_dir = "checkpoints"
    else:
        save_folder_dir = opts.save_folder_dir
    if not os.path.exists(save_folder_dir):
        os.makedirs(save_folder_dir, exist_ok=True)

    # Setting model
    train_data_by_modality = {}
    encoder_dim_list = []
    decoder_dim_list = []
    latent_dim = {}
    for i in modality:
        if i not in opts.decoder_dict.keys():
            raise ValueError("Decoder dict should have the same keys as encoder dict")
        else:
            current_modality_key = opts.modality_keys[i]
            if current_modality_key == "X":
                if isinstance(train_data.X,np.ndarray):
                    train_data_by_modality[i] = train_data.X.astype(dtype)
                elif isinstance(train_data.X,sp.csr_matrix):
                    train_data_by_modality[i] = train_data.X.toarray().astype(dtype)
                else:
                    raise ValueError("Unknown data type for modality {}, please check the data type. We support dataframe, numpy, andc csr matrix".format(i))
            else:
                if isinstance(train_data.obsm[current_modality_key],np.ndarray):
                    train_data_by_modality[i] = train_data.obsm[current_modality_key].astype(dtype)
                elif isinstance(train_data.obsm[current_modality_key],pd.DataFrame):
                    train_data_by_modality[i] = train_data.obsm[current_modality_key].to_numpy().astype(dtype)
                elif isinstance(train_data.obsm[current_modality_key],sp.csr_matrix):
                    train_data_by_modality[i] = train_data.obsm[current_modality_key].toarray().astype(dtype)
                else:
                    raise ValueError("Unknown data type for modality {}, please check the data type. We support dataframe, numpy, andc csr matrix".format(i))
            encoder_dim_list.append([train_data_by_modality[i].shape[1]]+opts.encoder_dict[i])
            decoder_dim_list.append(opts.decoder_dict[i]+[train_data_by_modality[i].shape[1]])
            latent_dim[i] = opts.encoder_dict[i][-1]
    modality_dict_encoder = get_dict_from_modality(modality,encoder_dim_list)
    modality_dict_decoder = get_dict_from_modality(modality,decoder_dim_list)
    classifier_interlayers_dims = opts.classifier_interlayers_dims["Classifier"]

    if opts.celltype is None:
        cell_type = ['celltype' for i in range(train_data.shape[0])]
    else:
        cell_type = train_data.obs[opts.celltype].to_list()

    patient_group = train_data.obs[opts.class_label].to_numpy()

    #if patient group is not numeric, we need to convert it to numeric label by label encoder
    if not np.issubdtype(patient_group.dtype, np.number):
        le = LabelEncoder()
        patient_group = le.fit_transform(patient_group)


    activation = nn.LeakyReLU()
    batchnorm = False
    dropout = 0.1
    dropout_classifier = 0

    #get the number of batch
    if opts.categorical_covariate is None:
        num_batch = 0
    else:
        batch = train_data.obs[opts.categorical_covariate].to_numpy()
        #if batch is not numeric, we need to convert it to numeric label by label encoder
        if not np.issubdtype(batch.dtype, np.number):
            le = LabelEncoder()
            batch = le.fit_transform(batch)
        #one hot encoding batch
        enc = OneHotEncoder(handle_unknown='ignore')
        batch = enc.fit_transform(batch.reshape(-1, 1)).toarray().astype(dtype)
        num_batch = batch.shape[1]



    print("num_class", num_class)
    print("num_batch", num_batch)
    print("latent_dim", latent_dim)
    print("modality_dict_encoder", modality_dict_encoder)
    print("modality_dict_decoder", modality_dict_decoder)
    print("classifier_interlayers_dims", classifier_interlayers_dims)
    print("modality", modality)
    if opts.categorical_covariate is None:
        train_set = [train_data_by_modality[i] for i in modality]+[patient_group]
        dataindex = slice(0, len(modality))
        batchindex = None
        patient_group_index = len(modality)

    else:
        #combine every running_utils related data to a list
        train_set = [train_data_by_modality[i] for i in modality]+[batch,patient_group]

        #create index to get data from train_set
        dataindex = slice(0, len(modality))
        batchindex = len(modality)
        patient_group_index = batchindex+1
    unique_celltype = np.unique(cell_type)
    #index = (dataindex, batchindex, patient_group_index)
    num_sample = train_set[0].shape[0]


    #create dataloader
    train_loader = create_dataloader(train_set, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True,
                                     create_data_loader=True)
    #create model
    model = AE_Classifier(modality_dict_encoder, modality_dict_decoder, classifier_interlayers_dims, num_batch,
                          num_class, activation=activation, batchnorm=batchnorm, dropout=dropout,
                          dropout_classifier=dropout_classifier)

    num_iter = int(num_sample * opts.iteraction_multiplier / batch_size)
    warm_up = opts.warm_up
    early_stop_tolerance_final_step = opts.early_stop_tolerance_final_step
    optimizer = Adam(model.parameters(), lr=opts.learning_rate)
    criterion = nn.MSELoss(reduction='mean')
    model.enable_AE_grad()
    model.to(device)
    model = pre_training(model, train_loader, optimizer, criterion, device, num_iter, batchindex, dataindex, warm_up,
                     early_stop_tolerance_final_step, save_folder_dir)


    #generate embedding after pretraining
    model.eval()
    model.cpu()
    model_input = get_input(train_set, dataindex)
    if batchindex is not None:
        model.set_batch_onehot(train_set[batchindex])
    else:
        model.set_batch_onehot(None)
    with torch.no_grad():
        _, emb_train = model(model_input, train_what="AE")
    emb_train = emb_train.detach().cpu().numpy()
    train_data.obsm["scSniper_emb"] = emb_train
    sc.pp.neighbors(train_data, use_rep="scSniper_emb")
    sc.tl.umap(train_data, min_dist=0.2)
    #save figure to save_folder_dir+"/pretraining.png"
    sc.settings.figdir = save_folder_dir
    sc.pl.umap(train_data, color=opts.celltype, save="pretraining_umap.png", show=False)
    print("Pretraining finished, please check the embedding in {}".format(save_folder_dir+"/pretraining.png"))

    ################Staring cell type specific running_utils################
    print("Pretraining Finished, Starting cell type specific running_utils")

    cell_type_list = []
    cell_type_loss_change = {}
    for current_cell_type in unique_celltype:
        print(current_cell_type)

        #indexing data for current cell type
        current_cell_type_index = train_data.obs[opts.celltype] == current_cell_type
        #current_cell_type_data = train_data[current_cell_type_index].copy()
        # current_cell_type_data_by_modality = {}
        # for i in modality:
        #     if opts.modality_keys[i] == "X":
        #         current_cell_type_data_by_modality[i] = train_set[i][current_cell_type_index]
        #     else:
        #         current_cell_type_data_by_modality[i] = train_set[i][current_cell_type_index]
        # if opts.categorical_covariate is None:
        #     current_cell_type_train_set = [current_cell_type_data_by_modality[i] for i in modality] + [train_set[patient_group_index][current_cell_type_index]]
        # else:
        #     current_cell_type_train_set = [current_cell_type_data_by_modality[i] for i in modality] + [train_set[batchindex][current_modality_key],train_set[patient_group_index][current_cell_type_index]]

        current_cell_type_train_set = [i[current_cell_type_index] for i in train_set]
        #create model
        model = AE_Classifier(modality_dict_encoder, modality_dict_decoder, classifier_interlayers_dims, num_batch,
                              num_class, activation=activation, batchnorm=batchnorm, dropout=dropout,
                              dropout_classifier=dropout_classifier)
        model.load_state_dict(torch.load(save_folder_dir+"/best_model.pt"))
        model.train()
        #Stage 1
        model.enable_AE_grad()
        model.to(device)
        batch_size = opts.batch_size_fine_tune
        lr = opts.learning_rate_fine_tune
        criterion = nn.MSELoss(reduction='mean')
        optimizer = Adam(model.parameters(), lr=lr)
        N_EPOCHS = opts.iteraction_fine_tune
        train_class_weight = get_class_weight(current_cell_type_train_set[-1])
        train_class_weight = torch.FloatTensor(train_class_weight).to(device)

        print("There are {} positive samples and {} negative samples".format(torch.sum(current_cell_type_train_set[-1] == 1),
                                                                             torch.sum(current_cell_type_train_set[-1] == 0)))
        if (torch.bincount(current_cell_type_train_set[-1]) == 0).any() or torch.bincount(current_cell_type_train_set[-1]).shape[
            0] < 2:
            print("Warning: class {} has 0 positive or negative sample".format(current_cell_type))
            print(torch.bincount(current_cell_type_train_set[-1]))
            print("Skip this class")
            print("=====================================")
            continue
        cell_type_list.append(current_cell_type)

        train_loader = create_dataloader(current_cell_type_train_set, batch_size=batch_size, num_workers=8, pin_memory=True,
                                         shuffle=True, create_data_loader=True)
        training_celltype_stage1(model, train_loader, optimizer, criterion, device, N_EPOCHS, batchindex, dataindex, warm_up,
                                         early_stop_tolerance_final_step,current_cell_type,save_folder_dir)
        model.cpu()
        model.load_state_dict(torch.load(save_folder_dir+"/best_model" + str(current_cell_type.replace("/", "_")) + ".pt"))

        #Stage 2
        model.enable_Classifier_grad()
        model.to(device)
        criterion_classification = FocalLoss(gamma=1, weights=train_class_weight, reduction='mean')
        optimizer = Adam(model.parameters(), lr=lr)
        training_celltype_stage2(model, train_loader, optimizer, criterion_classification, device, N_EPOCHS, batchindex, dataindex, warm_up,
                                            early_stop_tolerance_final_step,current_cell_type,patient_group_index,save_folder_dir)
        model.cpu()
        model.load_state_dict(torch.load(save_folder_dir+"/best_model" + str(current_cell_type.replace("/", "_")) + ".pt"))

        #Stage 3

        model.enable_all_grad()
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        training_celltype_stage3(model, train_loader, optimizer, criterion, device, N_EPOCHS, batchindex, dataindex, warm_up,
                                    early_stop_tolerance_final_step,current_cell_type,patient_group_index,criterion_classification,opts.lambd,save_folder_dir)

        model.cpu()
        model.load_state_dict(torch.load(save_folder_dir+"/best_model" + str(current_cell_type.replace("/", "_")) + ".pt"))

        #evaluation
        print("Evaluation for the cell type, starting")
        print("Please note that the evaluation will use the entire data in one batch, which may cause memory error")
        print("If you encounter memory error, please change to a GPU with more memory")
        print("Sorry for the inconvenience, or you can use CPU to run the evaluation, which will be slow")
        print("If you want to use CPU, please change the device_eval to cpu at line 306 in train.py")
        print("Future version will fix this issue")
        loss_change = {}
        loss_fun = [criterion_classification, criterion]




        #change this to cpu if you want to use cpu. i.e. device_eval = "cpu"
        device_eval = device



        model.eval()
        model.to(device_eval)

        eval_set = [i.to(device_eval) for i in current_cell_type_train_set]
        eval_loop(model, eval_set, loss_fun, batchindex, dataindex, patient_group_index, loss_change, opts.lambd)
        model.cpu()
        #rename key of loss_change to train_data_by_modality.keys()
        loss_change = {i:loss_change[j] for i,j in zip(modality,range(len(modality)))}
        cell_type_loss_change[current_cell_type] = loss_change
        print("Evaluation for the cell type, finished")
        print("=====================================")
    pickle.dump(cell_type_loss_change, open(save_folder_dir + "/cell_type_loss_change.pkl", "wb"))
    print("Training finished, saving the loss change.")
    print("Thank you for choosing scSniper. We hope you enjoyed it.")

if __name__ == '__main__':
    main()