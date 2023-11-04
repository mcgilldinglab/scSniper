import numpy as np
import torch
import re
import json
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

# ----------------------------------------------------------------------------
# From https://github.com/NVlabs/edm
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_dict_of_int_lists(dict_string):
    # First, we convert the string representation into a valid JSON string by replacing single quotes with double quotes
    # and ensuring lists are properly formatted as JSON arrays.
    dict_string = dict_string.replace("'", '"')
    dict_string = re.sub(r'(\w+):', r'"\1":', dict_string)  # Ensure keys are in double quotes
    dict_string = re.sub(r'(\d),', r'\1,', dict_string)  # Remove spaces after commas
    dict_string = re.sub(r',(\s*})', r'\1', dict_string)  # Remove any trailing commas
    dict_string = re.sub(r':(\d+)', r':[\1]', dict_string)  # Wrap single integers in lists
    dict_string = re.sub(r'\[([^\]]+)\]', lambda m: '[' + re.sub(r'(\d+)', r'\1', m.group(1)) + ']', dict_string)  # Ensure numbers in lists are properly formatted

    # Now, we attempt to decode the JSON string into a Python dictionary.
    try:
        dict_of_int_lists = json.loads(dict_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"String is not a valid dictionary representation: {e.msg}")

    # We ensure all values in the dictionary are lists of integers.
    for key, value in dict_of_int_lists.items():
        if not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' is not a list.")
        dict_of_int_lists[key] = [int(i) if isinstance(i, int) else int(i) for i in value]

    return dict_of_int_lists

def parse_dict_of_strings(dict_string):
    # Properly format the input for JSON
    # Replace single quotes with double quotes
    # Add double quotes around keys without quotes
    formatted_dict_string = re.sub(r"(\w+): '([^']*)'", r'"\1": "\2"', dict_string)
    formatted_dict_string = re.sub(r"(\w+): \"([^\"]*)\"", r'"\1": "\2"', formatted_dict_string)
    formatted_dict_string = re.sub(r'(\w+):(\w+)', r'"\1":"\2"', formatted_dict_string)

    #add double quotes around keys without quotes

    try:
        # Parse the string into a Python dictionary
        parsed_dict = json.loads(formatted_dict_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"String is not a valid dictionary representation: {e.msg}")

    # Validate that all values are strings
    for key, value in parsed_dict.items():
        if not isinstance(value, str):
            raise ValueError(f"Value for key '{key}' is not a string.")

    return parsed_dict
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
def get_input(data,index):
    if isinstance(index,slice):
        x = data[index]
    else:
        x = [data[index]]
    return x
def create_dataloader(data, batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=False, sampler=None,create_data_loader = True):
    if not create_data_loader:
        return_list = []
        for i in data:
            try:
                return_list.append(torch.from_numpy(i).cuda())
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

@torch.jit.script
def set_zeros(data: torch.Tensor,index: int)->torch.Tensor:
    data_copy = data.clone()
    data_copy[:,index] = 0
    return data_copy