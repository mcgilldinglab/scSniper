from tqdm import tqdm
import torch
import torch.nn as nn

def pre_training(model, train_loader, optimizer, criterion, device, num_iter, batch_index, dataindex, warm_up, early_stop_tolerance_final_step,save_folder_dir):
    best_val_loss = float('inf')
    for i in tqdm(range(num_iter)):
        input = next(train_loader)
        input = [i.to(device) if isinstance(i,torch.Tensor) else torch.from_numpy(i).to(device) for i in input]
        #training_part

        optimizer.zero_grad()
        model.train()
        model.set_batch_onehot(input[batch_index])
        x_re,emb_train = model(input[dataindex],train_what = "AE")
        train_loss = 0
        for j in range(len(x_re)):
            train_loss += criterion(x_re[j],input[dataindex][j])
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