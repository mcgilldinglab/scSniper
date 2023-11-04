import click
import numpy as np
import pickle
import os
import pandas as pd
import scanpy as sc
from running_utils.misc import parse_dict_of_int_lists, parse_dict_of_strings
@click.command()
@click.option('--data', help='Path to the dataset (H5AD)', metavar='H5AD', type=str, required=True)
@click.option('--output', help='Path to the output file', metavar='OUTPUT', type=str, required=True)
@click.option('--result_file', help='Path to the result file', metavar='PKL', type=str, required=True)
@click.option('--modality_keys', help='h5ad obsm keys to each modality, if you want to use .X, please input "X".Example: {RNA:X,ADT:Antibody Capture}}', metavar='DICT', type=parse_dict_of_strings, default=None,required=True)
@click.option('--num_features', help='Dict format. Number of Feature you want for each modality.', metavar='DICT', type=parse_dict_of_int_lists, default="{RNA:[100],ADT:[50}",required=True)

def main(**kwargs):
    data = kwargs['data']
    output = kwargs['output']
    result_file = kwargs['result_file']
    modality_keys = kwargs['modality_keys']
    features = kwargs['num_features']
    print("Loading data...")
    data = sc.read_h5ad(data)
    print("Loading result...")
    result = pickle.load(open(result_file, "rb"))
    print("We assume that gene name is in var.index")
    print("We assume that each obsm's column name is the feature name of that modality")
    print("Loading modality keys...")
    print("Loading feature names...")

    modality_names = list(modality_keys.keys())
    feature_names = {}
    for modality_name in modality_names:
        if modality_name not in features:
            raise ValueError(f"Modality name '{modality_name}' is not in features.")
        if modality_keys[modality_name] == 'X':
            feature_names[modality_name] = data.var.index.tolist()
        else:
            feature_names[modality_name] = data.obsm[modality_keys[modality_name]].columns.tolist()

    print("Calulating feature importance...")
    for current_cell_type in result.keys():
        # create output folder
        current_cell_type_out_folder = os.path.join(output, current_cell_type)
        if not os.path.exists(current_cell_type_out_folder):
            os.makedirs(current_cell_type_out_folder, exist_ok=True)
        else:
            raise ValueError(f"Output folder {current_cell_type_out_folder} already exists.")

        print(f"Current cell type: {current_cell_type}")
        for modality_name in modality_names:
            modality_feature = feature_names[modality_name]
            modality_feature_importance = result[current_cell_type][modality_name]
            modality_feature_importance = modality_feature_importance[1:] - modality_feature_importance[0]
            topx = np.argsort(modality_feature_importance)[-features[modality_name][0]:]
            print(f"Modality: {modality_name}")

            feature_index = pd.Index(modality_feature)
            feature_index[topx].to_series().to_csv(current_cell_type_out_folder+"/"+modality_name+".txt",header=False,index=False,sep="\n")
        print("Now finding joint biomarkers...")
        all_features = []
        all_feature_names = []
        for modality_name in modality_names:
            modality_feature_importance = result[current_cell_type][modality_name]
            modality_feature_importance = modality_feature_importance[1:] - modality_feature_importance[0]
            all_features.append(modality_feature_importance)
            all_feature_names.extend(feature_names[modality_name])
        all_features = np.concatenate(all_features,axis=0)
        print("By default, we use the sum of number of features you want for each modality as the number of joint biomarkers.")
        sum_of_num_features = sum([i[0] for i in features.values()])
        topx = np.argsort(all_features)[-sum_of_num_features:]
        feature_index = pd.Index(all_feature_names)
        feature_index[topx].to_series().to_csv(current_cell_type_out_folder+"/Joint_biomarkers.txt",header=False,index=False,sep="\n")

    print("Done!")
    print("Please check your output folder.")
    print("Thank you for choosing scSniper. We hope you enjoyed it.")


if __name__ == '__main__':
    main()