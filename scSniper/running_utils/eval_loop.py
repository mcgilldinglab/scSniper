import torch
from .misc import set_zeros
from tqdm import tqdm
from .misc import get_input
import numpy as np
def eval_loop(model, train_set_thisADT, loss_fun, batch_index, dataindex,patient_cat_index,acc_total,lamb):
    model.eval()
    with torch.no_grad():
        train_patient_cat = train_set_thisADT[patient_cat_index]
        if batch_index is not None:
            model.set_batch_onehot(train_set_thisADT[batch_index])
        else:
            model.set_batch_onehot(None)
        x_re, emb_train, output = model(train_set_thisADT[dataindex], train_what="all")
        train_loss = (1-lamb) * loss_fun[0](output, train_patient_cat)

        model_input = get_input(train_set_thisADT, dataindex)

        for i in range(len(model_input)):
            train_loss += lamb * loss_fun[1](x_re[i], model_input[i])
        for i in range(len(model_input)):
            acc_total[i] = [train_loss.item()]


        for modality_index in range(len(model_input)):
            print("Modality {} eval starting".format(modality_index))
            for i in tqdm(range(model_input[modality_index].shape[1])):
                current_modality_with_one_zero = set_zeros(model_input[modality_index], i)
                model_input_with_one_zero = model_input.copy()
                model_input_with_one_zero[modality_index] = current_modality_with_one_zero
                x_re, emb_train, output = model(model_input_with_one_zero, train_what="all")
                train_loss = (1-lamb) * loss_fun[0](output, train_patient_cat)
                for j in range(len(model_input)):
                    train_loss += lamb * loss_fun[1](x_re[j], model_input_with_one_zero[j])
                acc_total[modality_index].append(train_loss.item())
        for i in range(len(model_input)):
            acc_total[i] = np.array(acc_total[i])
    return acc_total