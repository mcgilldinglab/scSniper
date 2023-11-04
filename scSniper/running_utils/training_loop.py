from tqdm import tqdm
from .misc import get_input
import torch

def pre_training(model, train_loader, optimizer, criterion, device, num_iter, batch_index, dataindex, warm_up, early_stop_tolerance_final_step,save_folder_dir):
    best_val_loss = float('inf')
    for i in tqdm(range(num_iter)):
        input = next(train_loader)
        input = [i.to(device) if isinstance(i,torch.Tensor) else torch.from_numpy(i).to(device) for i in input]

        optimizer.zero_grad()
        model.train()
        if batch_index is not None:
            model.set_batch_onehot(input[batch_index])
        else:
            model.set_batch_onehot(None)
        model_input = get_input(input,dataindex)
        x_re,emb_train = model(model_input,train_what = "AE")
        train_loss = 0
        for j in range(len(x_re)):
            train_loss += criterion(x_re[j],model_input[j])
        train_loss.backward()
        optimizer.step()

        if train_loss < best_val_loss or i <= warm_up:
            best_val_loss = train_loss
            torch.save(model.state_dict(), save_folder_dir+"/best_model.pt")
            count = 0
        else:
            count += 1
            if count >= early_stop_tolerance_final_step:
                print("AE early stopping at epoch {}".format(i+1))
                return model
                break
    return model

def training_celltype_stage1(model, train_loader,optimizer,criterion, device, N_iter, batch_index,dataindex, warm_up, early_stop_tolerance_final_step,current_cell_type,save_folder_dir):
    best_val_loss = float('inf')
    for i in tqdm(range(N_iter)):
        model.train()
        optimizer.zero_grad()
        input = next(iter(train_loader))

        input = [i.to(device) for i in input]
        if batch_index is not None:
            model.set_batch_onehot(input[batch_index])
        else:
            model.set_batch_onehot(None)
        model_input = get_input(input, dataindex)
        x_re, emb_train_this, output_this = model(model_input, train_what="all")

        train_loss = 0
        for j in range(len(x_re)):
            train_loss += criterion(x_re[j], model_input[j])


        train_loss.backward()
        optimizer.step()
        train_loss = train_loss.item()

        if train_loss < best_val_loss or i <= warm_up:
            best_val_loss = train_loss
            current_cell_type_name = current_cell_type.replace("/", "_")
            torch.save(model.state_dict(), save_folder_dir+"/best_model" + str(current_cell_type_name) + ".pt")
            count = 0
        else:
            count += 1
            if count >= early_stop_tolerance_final_step:
                print("AE early stopping at epoch {}".format(i + 1))
                break

def training_celltype_stage2(model, train_loader,optimizer,criterion_classification, device, N_iter, batch_index,dataindex, warm_up, early_stop_tolerance_final_step,current_cell_type,patient_cat_index,save_folder_dir):
    best_val_loss = float('inf')
    for i in tqdm(range(N_iter)):
        model.train()
        optimizer.zero_grad()
        input = next(iter(train_loader))

        input = [i.to(device) for i in input]
        if batch_index is not None:
            model.set_batch_onehot(input[batch_index])
        else:
            model.set_batch_onehot(None)
        model_input = get_input(input, dataindex)
        x_re, emb_train_this, output_this = model(model_input, train_what="all")
        train_loss = criterion_classification(output_this, input[patient_cat_index])

        train_loss.backward()
        optimizer.step()
        train_loss = train_loss.item()

        if train_loss < best_val_loss or i <= warm_up:
            best_val_loss = train_loss
            current_cell_type_name = current_cell_type.replace("/", "_")
            torch.save(model.state_dict(), save_folder_dir+"/best_model" + str(current_cell_type_name) + ".pt")
            count = 0
        else:
            count += 1
            if count >= early_stop_tolerance_final_step:
                print("AE early stopping at epoch {}".format(i + 1))
                break


def training_celltype_stage3(model, train_loader,optimizer,criterion, device, N_iter, batch_index,dataindex, warm_up, early_stop_tolerance_final_step,current_cell_type,patient_cat_index,criterion_classification,lamb,save_folder_dir):
    best_val_loss = float('inf')
    for i in tqdm(range(N_iter)):
        model.train()
        optimizer.zero_grad()
        input = next(iter(train_loader))

        input = [i.to(device) for i in input]
        if batch_index is not None:
            model.set_batch_onehot(input[batch_index])
        else:
            model.set_batch_onehot(None)
        model_input = get_input(input, dataindex)
        x_re, emb_train_this, output_this = model(model_input, train_what="all")
        train_loss = criterion_classification(output_this, input[patient_cat_index]) * (1-lamb)
        for j in range(len(x_re)):
            train_loss += criterion(x_re[j], model_input[j]) * lamb

        train_loss.backward()
        optimizer.step()
        train_loss = train_loss.item()

        if train_loss < best_val_loss or i <= warm_up:
            best_val_loss = train_loss
            current_cell_type_name = current_cell_type.replace("/", "_")
            torch.save(model.state_dict(), save_folder_dir+"/best_model" + str(current_cell_type_name) + ".pt")
            count = 0
        else:
            count += 1
            if count >= early_stop_tolerance_final_step:
                print("AE early stopping at epoch {}".format(i + 1))
                break