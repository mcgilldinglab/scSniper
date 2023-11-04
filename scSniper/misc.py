import numpy as np
import torch

def get_dict_from_modality(modality_list,modality_dim_list):
    modality_dict = {}
    for i,modality in enumerate(modality_list):
        modality_dict[modality] = modality_dim_list[i]
    return modality_dict

def set_seed(seed):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


def get_class_weight(y):
    num_samples_per_class = np.bincount(y)
    weight = 1. / num_samples_per_class
    weight = weight / np.sum(weight)
    return weight


def get_data(data,index):
    if index[1] is None:
        batch_label = None
    else:
        batch_label = data[index[1]]
    if isinstance(index[0],slice):
        x_rna = data[index[0]]
    else:
        x_rna = [data[index[0]]]
    return x_rna, batch_label,data[index[2]]

def create_dataloader(data, batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=False, sampler=None,create_data_loader = True):
    if not create_data_loader:
        return_list = []
        for i in data:
            try:
                return_list.append(torch.from_numpy(i).cuda)
            except:
                return_list.append(i.cuda())
        return return_list

    for i in range(len(data)):
        try:
            data[i] = data[i].toarray()
        except:
            pass
        try:
            data[i] = torch.from_numpy(data[i])
        except:
            data[i] = data[i].detach().cpu()
    dataset = torch.utils.data.TensorDataset(*data)
    inf_sampler = InfiniteSampler(dataset, shuffle=shuffle)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
                                              num_workers=num_workers, pin_memory=pin_memory, sampler=inf_sampler)
    data_loader = iter(data_loader)
    return data_loader