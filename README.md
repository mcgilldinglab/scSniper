# 🎯 scSniper: Single-cell Deep Neural Network-basd Identification of Prominent Biomarkers
*A novel method for biomarker discovery that leverages deep neural network sensitivity analysis and an attention mechanism to select pivotal multi-omics biomarkers without relying on differential expression.*

This repository contains the official implementation of the paper:
> __scSniper: Single-cell Deep Neural Network-basd Identification of Prominent Biomarkers__
> [Mingyang Li],[Yanshuo Chen],[Jun Ding]
> 
## Overview
<p align="center">
  <img src=images/scSniperLogo.png alt="GitHub Logo" width="500" height="300">
</p>
scSniper introduces a novel approach to biomarker discovery, leveraging deep neural network sensitivity analysis to pinpoint key gene biomarkers beyond traditional differential expression methods. It stands out by utilizing a mimetic attention mechanism, which allows for the integration of multi-omic data, highlighting critical biomarkers across genomics, proteomics, and metabolomics. This mechanism prioritizes important multi-omic features, enabling a comprehensive analysis that other single-omic focused tools may miss.The tool innovatively merges a disease classifier and an autoencoder, producing joint cell embeddings that represent disease-specific multi-omic profiles, improving single-cell resolution clustering and biomarker identification accuracy. scSniper's method represents a significant leap in biomarker discovery, offering a more integrative and precise approach to understanding complex diseases.

## Setup
### Requirements
scSniper requires Python 3.9 or higher. The following packages are required:
```bash
pip install -r requirements.txt
```
By default, torch will be installed with CUDA support only on Linux. If you are using Windows or macOS. Instead of the above command, please install the appropriate PyTorch package for your system from [here](https://pytorch.org/get-started/locally/). Then run the following command instead:
```bash
pip install -r requirements_non_linux.txt
```

## Data preprocessing
We assumed your data are ready preprocessed. If not, we recommand:

For RNA modality: use `sc.pp.filter_genes`, `sc.pp.filter_cells`, `sc.pp.normalize_total`, `sc.pp.log1p` to preprocess your data. You can view the [scanpy tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html) for more details

For ADT modality: use `CTL` normalization. You can use muon package to do this. You can view the [muon tutorial](https://muon.readthedocs.io/en/latest/tutorials.html) for more details.

For ATAC modality: filter peak first and binarize your data.

For all other modalities: we believe the preprocessing steps are similar to the above modalities. If you have any questions, please open an issue.

We also assumed non-RNA modality is stored in `obsm` of the AnnData object as a dataframe where the column names are the feature names and the row names are the cell names. The row order should be the same as the RNA modality. If you have any questions, please open an issue.

## Training
Required arguments:
* `--data`: Path to the data file. We assumed it's a H5AD file.
* `--modality_keys`: The keys of the modalities in the data file. For example, if the data file contains RNA and ATAC data, then the keys should be `X` for RNA, and the obsm key for ATAC. Example: `--modality_keys="{RNA:"X","ATAC":"ATAC_data"}"`.
* `--class_label`: The key of the disease class label in the data file.  Example: `--label_key=patient_cat`.
* `--output_path`: Path to the output folder.
* `--num_class`: The number of disease classes. Example: `--num_class=2`.
* `--encoder_dict`: The output dimensions of the encoder layers. The last input dimension is the latent dimension of the modality. Input dimensions are automatically calculated. Example `--encoder_dict="{RNA:[128,128,64],ATAC:[128,64,32]}"`.
* `--decoder_dict`: The output dimensions of the decoder layers. The last input dimension is the latent dimension of the modality. Input dimensions are automatically calculated. Example `--decoder_dict="{RNA:[128,128],ATAC:[64,64,128]}"`.

Optionally, you can specify the following arguments:
* `--batch_size`: The batch size. Default: `--batch_size=128`.
* `--learning_rate`: The learning rate. Default: `--lr=1e-3`.
* `--categorical_covariate`: The categorical covariate, such as batch_label. Default: `--categorical_covariate=None`.
* `--classifier_interlayers_dims`: The dimensions of the classifier interlayers. Default: `--classifier_interlayers_dims="{Classifier:[32,10]}"`.

We listed important arguments here. You can find all arguments in `train.py` with `python train.py --help`.
```bash
python train.py --your_arguments=your_values ...
```


The program will automatically save the result in a pickle file named `cell_type_loss_change.pkl` to the output folder. You can run `python eval.py --help` to find biomarkers as illustrated below.
## Evaluation
Required arguments:
* `--data`: Path to the data file. We assumed it's a H5AD file.
* `--output`: Path to the output folder to save the biomarkers.
* `--modality_keys`: The keys of the modalities in the data file. For example, if the data file contains RNA and ATAC data, then the keys should be `X` for RNA, and the obsm key for ATAC. Example: `--modality_keys="{RNA:"X","ATAC":"ATAC_data"}"`.
* `--result_file`: The path to the pickle file generated by `train.py`. Example: `--result_file=cell_type_loss_change.pkl`.
* `--num_features`: The number of biomarkers to select per modality. Example: `--num_features="{RNA:100,ATAC:100}"`.
```bash
python eval.py --your_arguments=your_values ...
```
Always use `python eval.py --help` to see all the arguments and their default values.

This will create folders for each `cell_type` in the output folder. The folder contains the following files:
* `modality_1`: The biomarkers for modality 1. A text file where each line is a biomarker.
* ...
* `modality_n`: The biomarkers for modality n. A text file where each line is a biomarker.
* `Joint`: The cross-modality biomarkers. A text file where each line is a biomarker.
## To Do
- [ ] Add distributed training (more than one GPU)
- [ ] Add option to use different optimizers

## Acknowledgement
This work was funded in part by grants awarded to [JD]. We gratefully acknowledge the support from the Canadian Institutes of Health Research (CIHR) under Grant Nos. PJT-180505; the Funds de recherche du Québec - Santé (FRQS) under Grant Nos. 295298 and 295299; the Natural Sciences and Engineering Research Council of Canada (NSERC) under Grant No. RGPIN2022-04399; and the Meakins-Christie Chair in Respiratory Research.
## Citation
