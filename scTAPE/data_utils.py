import os
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn import preprocessing
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset

root_data_folder = './data/'
raw_data_folder = os.path.join(root_data_folder, 'raw_dat')

# CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'CCLE_expression.csv')
ccle_sample_file = os.path.join(ccle_folder, 'sample_info.csv')

# GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_raw_target_file = os.path.join(gdsc_folder, 'gdsc_ic50flag.csv')
gdsc_sample_file = os.path.join(gdsc_folder, 'gdsc_cell_line_annotation.csv')


class AlignedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]


def get_unlabeled_dataloaders(args, ccle_data, sc_data):
    """
    CCLE as source domain, thus s_dataloaders
    single-cell as target domain, thus t_dataloaders
    """
    ccle_sample_info_df = pd.read_csv("processed_data/bulk_ccle_sample_info.csv", index_col=0)
    sc_sample_info_df = pd.read_csv("processed_data/sc_ccle_sample_info.csv", index_col=1, low_memory=False)

    # scaler1 = preprocessing.StandardScaler()
    # scaler2 = preprocessing.StandardScaler()

    # # 分别对两个数据集进行标准化
    # ccle_data_X = scaler1.fit_transform(ccle_data.X)
    # sc_data_X = scaler2.fit_transform(sc_data.X)

    # # 合并标准化后的数据
    # combined_data = np.vstack((ccle_data_X, sc_data_X))

    # # 对合并后的数据再次标准化
    # scaler_combined = preprocessing.StandardScaler()
    # scaler_combined.fit(combined_data)
    # ccle_data_X = scaler_combined.transform(ccle_data_X)
    # sc_data_X = scaler_combined.transform(sc_data_X)

    ccle_data_df = pd.DataFrame(ccle_data.X, index=ccle_data.obs_names, columns=ccle_data.var_names)
    sc_data_df = pd.DataFrame(sc_data.X, index=sc_data.obs_names, columns=sc_data.var_names)
        
    ccle_sample_info_df = ccle_sample_info_df.loc[ccle_data_df.index]
    sc_sample_info_df = sc_sample_info_df.loc[sc_data_df.index]

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_data_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)

    to_split_ccle_df = ccle_data_df[~ccle_data_df.index.isin(excluded_ccle_samples)]

    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=0.2, stratify=
                                                   ccle_sample_info_df.loc[to_split_ccle_df.index].primary_disease)
    
    test_ccle_df = pd.concat([test_ccle_df, ccle_data_df.loc[excluded_ccle_samples]])           
    train_ccle_df_v, test_ccle_df_v = subvector(args.gap_num, train_ccle_df.values), subvector(args.gap_num, test_ccle_df.values)
    
    train_ccle_dataset_v = TensorDataset(torch.from_numpy(train_ccle_df_v.astype('float32')))
    train_ccle_dataset_i = TensorDataset(torch.from_numpy(train_ccle_df.values.astype('float32')))

    test_ccle_dataset_v = TensorDataset(torch.from_numpy(test_ccle_df_v.astype('float32')))
    test_ccle_dataset_i = TensorDataset(torch.from_numpy(test_ccle_df.values.astype('float32')))

    # single cell data processing
    train_sc_df, test_sc_df = train_test_split(sc_data_df, test_size=0.2, stratify=sc_sample_info_df['Cancer_type'])
    train_sc_df_v, test_sc_df_v = subvector(args.gap_num, train_sc_df.values), subvector(args.gap_num, test_sc_df.values)

    train_sc_dataset_v = TensorDataset(torch.from_numpy(train_sc_df_v.astype('float32')))
    train_sc_dataset_i = TensorDataset(torch.from_numpy(train_sc_df.values.astype('float32')))

    test_sc_dataset_v = TensorDataset(torch.from_numpy(test_sc_df_v.astype('float32')))
    test_sc_dataset_i = TensorDataset(torch.from_numpy(test_sc_df.values.astype('float32')))
    
    train_ccle_aligned = AlignedDataset(train_ccle_dataset_v, train_ccle_dataset_i)
    train_ccle_dataloader = DataLoader(train_ccle_aligned, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=1)

    test_ccle_aligned = AlignedDataset(test_ccle_dataset_v, test_ccle_dataset_i)
    test_ccle_dataloader = DataLoader(test_ccle_aligned, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    train_sc_aligned = AlignedDataset(train_sc_dataset_v, train_sc_dataset_i)
    train_sc_dataloader = DataLoader(train_sc_aligned, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=1)

    test_sc_aligned = AlignedDataset(test_sc_dataset_v, test_sc_dataset_i)
    test_sc_dataloader = DataLoader(test_sc_aligned, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    return [train_ccle_dataloader, test_ccle_dataloader], [train_sc_dataloader, test_sc_dataloader]


def get_labeled_PRISM_dataloader(gex_features_df, drug, batch_size, gam_num, seed, threshold, measure):

    drugs_to_keep = [drug.lower()]
    PRISM_response_df = pd.read_csv('data/raw_dat/PRISM/PRISM_AUC_values.csv', index_col=0)
    PRISM_response_df.index.name = None
    PRISM_response_df.loc[:, 'DRUG_NAME'] = PRISM_response_df['DRUG_NAME'].str.lower()

    # PRISM_sample_df = pd.read_csv('data/raw_dat/PRISM/secondary-screen-cell-line-info.csv', index_col=0)
    # ccle_sample_info = pd.read_csv(ccle_sample_file, index_col=0)
    # ccle_sample_info.index.name = None
    # ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]

    ccle_target_df = PRISM_response_df[PRISM_response_df['DRUG_NAME'] == drugs_to_keep[0]]
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if measure == 'Z_SCORE':
        threshold = 0.0

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples]['Z_SCORE'] < threshold).astype('int')

    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    x_train, x_test, y_train, y_test = train_test_split(ccle_labeled_feature_df, ccle_labels, test_size=0.2, 
                                                            random_state=seed, stratify=ccle_labels)
    
    class_sample_count = np.array([Counter(ccle_labels)[0] / len(ccle_labels), Counter(ccle_labels)[1] / len(ccle_labels)])
    weight = 1. / class_sample_count
    print("GDSC: Num of resistant and Num of Sensitive = {}".format(Counter(ccle_labels.values)))

    # upsampling source domain training set which is unbalanced
    samples_weight = np.array([weight[t] for t in y_train.values])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    train_labeled_ccle_df, test_labeled_ccle_df = x_train.values, x_test.values
    train_ccle_labels, test_ccle_labels = y_train.values, y_test.values
    
    train_labeled_ccle_df, test_labeled_ccle_df = subvector(gam_num, train_labeled_ccle_df), subvector(gam_num, test_labeled_ccle_df)

    train_labeled_ccle_dateset = TensorDataset(torch.from_numpy(train_labeled_ccle_df.astype('float32')),
                                                torch.from_numpy(train_ccle_labels))
    
    test_labeled_ccle_dataset = TensorDataset(torch.from_numpy(test_labeled_ccle_df.astype('float32')),
                                                torch.from_numpy(test_ccle_labels))
    
    train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset, batch_size=batch_size, sampler=sampler)

    test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_dataset, batch_size=batch_size, shuffle=True)

    yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader


def get_labeled_ccle_dataloader(gex_features_df, drug, batch_size, gam_num, seed, threshold, measure):

    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(gdsc_target_file1)
    gdsc2_response = pd.read_csv(gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measure]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measure]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()
    
    if measure == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measure] = np.exp(gdsc1_sensitivity_df[measure])
        gdsc2_sensitivity_df.loc[:, measure] = np.exp(gdsc2_sensitivity_df[measure])
    
    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measure, index='COSMIC_ID', columns='DRUG_NAME')
    
    ccle_sample_info = pd.read_csv(ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if measure == 'Z_SCORE':
        threshold = 0.0

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')

    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    x_train, x_test, y_train, y_test = train_test_split(ccle_labeled_feature_df, ccle_labels, test_size=0.2, 
                                                            random_state=seed, stratify=ccle_labels)
    
    class_sample_count = np.array([Counter(ccle_labels)[0] / len(ccle_labels), Counter(ccle_labels)[1] / len(ccle_labels)])
    weight = 1. / class_sample_count
    print("GDSC: Num of resistant and Num of Sensitive = {}".format(Counter(ccle_labels.values)))

    # upsampling source domain training set which is unbalanced
    samples_weight = np.array([weight[t] for t in y_train.values])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    train_labeled_ccle_df, test_labeled_ccle_df = x_train.values, x_test.values
    train_ccle_labels, test_ccle_labels = y_train.values, y_test.values
    
    train_labeled_ccle_df, test_labeled_ccle_df = subvector(gam_num, train_labeled_ccle_df), subvector(gam_num, test_labeled_ccle_df)

    train_labeled_ccle_dateset = TensorDataset(torch.from_numpy(train_labeled_ccle_df.astype('float32')),
                                                torch.from_numpy(train_ccle_labels))
    
    test_labeled_ccle_dataset = TensorDataset(torch.from_numpy(test_labeled_ccle_df.astype('float32')),
                                                torch.from_numpy(test_ccle_labels))
    
    train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset, batch_size=batch_size, sampler=sampler)

    test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_dataset, batch_size=batch_size, shuffle=True)

    yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader


def get_labeled_dataloader_generator(args, ccle_df):
    """
    sensitive (responder): 1
    resistant (non-responder): 0
    """
    # mmscaler1 = preprocessing.StandardScaler()
    # ccle_df_scaler = mmscaler1.fit_transform(ccle_df.X)
    # ccle_df_X = ccle_df_scaler
    
    sc_data = sc_recipe(drug=args.drug, gse_id=args.sc_data, var_names=ccle_df.var_names, args=args)

    # mmscaler2 = preprocessing.StandardScaler()
    # sc_data_scaler = mmscaler2.fit_transform(sc_data.X)
    # sc_data.X = sc_data_scaler
    
    # combined_data = np.vstack((ccle_df_scaler, sc_data_scaler))
    # scaler_combined = preprocessing.StandardScaler()
    # scaler_combined.fit(combined_data)

    # ccle_df_X = scaler_combined.transform(ccle_df_scaler)
    # sc_data_X = scaler_combined.transform(sc_data_scaler)
    # sc_data.X = sc_data_X

    ccle_data_df = pd.DataFrame(ccle_df.X, index=ccle_df.obs_names, columns=ccle_df.var_names)

    # get_labeled_ccle_dataloader, get_labeled_PRISM_dataloader
    ccle_labeled_dataloader = get_labeled_PRISM_dataloader(gex_features_df=ccle_data_df,
                                                          drug=args.drug,
                                                          batch_size=args.batch_size,
                                                          gam_num=args.gap_num,
                                                          seed=args.seed,
                                                          threshold=args.threshold,
                                                          measure=args.measurement)

    sc_labeled_dataloader = get_labeled_sc_dataloader(adata=sc_data,
                                                      batch_size=args.batch_size,
                                                      gap_num=args.gap_num,
                                                      task=args.task)
    
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, sc_labeled_dataloader


def get_labeled_sc_dataloader(adata, batch_size, gap_num, task):
    if task == 'patient':
        label = adata.obs
        sc_data = subvector(gap_num, adata.X)
        labeled_sc_dataset = TensorDataset(torch.from_numpy(sc_data.astype('float32')))
        labeled_sc_dataloader = DataLoader(labeled_sc_dataset, batch_size=batch_size)
        labeled_sc_dataloader = (labeled_sc_dataloader, label)
        
    elif task == 'cell_line':
        label = adata.obs['response'].values
        print("single cell: Num of resistant and Num of Sensitive = {}".format(Counter(label)))
        sc_data = subvector(gap_num, adata.X)
        labeled_sc_dataset = TensorDataset(torch.from_numpy(sc_data.astype('float32')), torch.from_numpy(label))
        labeled_sc_dataloader = DataLoader(labeled_sc_dataset, batch_size=batch_size)        
        
    return labeled_sc_dataloader


def sc_recipe(drug, gse_id, var_names, args):
    if args.task == 'patient': 
        sc_data_path = os.path.join('data/scdata/sc_patients/', gse_id, gse_id + '_subClone_fill_mean.h5ad')
        adata = sc.read_h5ad(sc_data_path)
        adata = adata[:, var_names].copy()
        # sc.pp.normalize_total(adata)
        # sc.pp.log1p(adata)

    elif args.task == 'cell_line':

        # sc_data_path = os.path.join('data/scdata/sc_lines/Bortezomib_processed.h5ad')
        # adata = sc.read_h5ad(sc_data_path)
        # existing_mask = np.isin(var_names, adata.var_names)
        # existing_genes = var_names[existing_mask]

        sc_data_path = os.path.join('data/scdata/sc_lines/', gse_id + '_' + drug + '_anndata.h5ad')
        adata = sc.read_h5ad(sc_data_path)
        adata = adata[:, var_names].copy()
        
        if gse_id not in ['GSE108383_451Lu']:
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
        else:
            sc.pp.log1p(adata)

    return adata


# Converting the gene expression matrix into sub-vectors
def subvector(gap, exp_mat):
    # (n_cells, n_genes) -> (n_cells, gap_num, gap)  gap_num = int(gene_num / gap) + 1
    X = exp_mat  # getting the gene expression matrix
    single_cell_list = []
    for single_cell in X:
        feature = []
        length = len(single_cell)
        # spliting the gene expression vector into some sub-vectors whose length is gap
        for k in range(0, length, gap):
            if (k + gap <= length):
                a = single_cell[k:k + gap]
            else:
                a = single_cell[length - gap:length]
            # scaling each sub-vectors 
            # a = preprocessing.scale(a)
            feature.append(a)
        feature = np.asarray(feature)
        single_cell_list.append(feature)
    
    single_cell_list = np.asarray(single_cell_list) #(n_cells, gap_num, gap)
    
    return single_cell_list


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    # else:
    #     print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def valid_seed():
    print("After reseeding:")
    print("Python random number:", random.random())
    print("NumPy random number:", np.random.rand())
    print("PyTorch random number (CPU):", torch.rand(1))
    if torch.cuda.is_available():
        print("PyTorch random number (GPU):", torch.cuda.FloatTensor(1).uniform_())