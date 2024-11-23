import sys
import os
import pandas as pd
import allel
from scripts import utils, stat, genom_pca


def log_step(message, char='-', width=80):
    separator = char * width
    print(f'\n{separator}\n{message}\n{separator}')


def get_train_dataset(
    vcf_file_path=r'..\data\genotypes.vcf',
    output_csv_file_path=r'..\data\parsed_vcf.csv',
    h5_file_path=r'..\data\genotypes.h5',
    weather_data_path=r'..\data\weather_by_year_v2.csv'
):
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

    log_step('Initializing file paths')
    VCF_FILE_PATH = vcf_file_path
    OUTPUT_CSV_FILE_PATH = output_csv_file_path
    H5_FILE_PATH = h5_file_path
    WEATHER_DATA_PATH = weather_data_path

    log_step('Parsing VCF to CSV')
    utils.parse_vcf_2_csv(VCF_FILE_PATH, OUTPUT_CSV_FILE_PATH)

    log_step('Loading parsed VCF CSV')
    df = pd.read_csv(r'..\data\parsed_vcf.csv')

    log_step('Extracting comprehensive genomic features')
    result = stat.extract_comprehensive_genomic_features(df)

    log_step('Dropping static columns')
    static_columns = [col for col in result.columns if result[col].nunique() == 1]
    new_result = result.drop(columns=static_columns, axis=1)

    log_step('Converting VCF to HDF5')
    allel.vcf_to_hdf5(
        input=VCF_FILE_PATH,
        output=H5_FILE_PATH,
        overwrite=True
    )

    log_step('Performing LD pruning and PCA')
    LD_PRUNE_DICT = dict(size=100, step=20, threshold=.1, n_iter=1)
    PCA_PARAMETERS_DICT = dict(n_components=15, scaler='patterson')

    chromo_dict = genom_pca.parse_genotype_by_chromo(H5_FILE_PATH)
    feature_dict = genom_pca.get_pca_vectors(chromo_dict, PCA_PARAMETERS_DICT, False, True, LD_PRUNE_DICT)

    log_step('Creating PCA DataFrame')
    df_list = []
    for key, value in feature_dict.items():
        feature_vector = value['feature_vector']
        sample_df = pd.DataFrame({
            f'Sample_{i+1}': [feature_vector[i]] for i in range(feature_vector.shape[0])
        })
        sample_df['Chroma'] = key
        df_list.append(sample_df)

    PCA_df = pd.concat(df_list, ignore_index=True)
    PCA_df = PCA_df.T
    PCA_df.columns = PCA_df.iloc[-1]
    PCA_df.to_csv(r'../data/20_chromas+KZ.csv')

    log_step('Resetting PCA DataFrame index')
    PCA_df = PCA_df.reset_index().drop('index', axis=1)

    log_step('Merging PCA results with genomic features')
    new_result_w_pca20 = pd.concat([new_result, PCA_df], axis=1)

    log_step('Loading and merging phenotype and vegetation data')
    ph = pd.read_csv(r'../data/phenotypes.tsv', sep='\t')
    vg = pd.read_csv(r'../data/vegetation.tsv', sep='\t')
    ph = ph.rename(columns={'sample': 'Sample'})
    vg = vg.rename(columns={'sample': 'Sample'})
    vg_w_ph = pd.merge(vg, ph, right_on='Sample', left_on='Sample')

    train = pd.merge(new_result_w_pca20, vg_w_ph, right_on='Sample', left_on='Sample')

    log_step('Melting data by years')
    id_vars = [
        'Sample', '0/0_homozygous_count', '0/1_homozygous_count',
        '1/1_homozygous_count', './._homozygous_count', '1/2_homozygous_count',
        '0/2_homozygous_count', '2/2_homozygous_count', '1/3_homozygous_count',
        '2/3_homozygous_count', '0/3_homozygous_count', '3/3_homozygous_count',
        'chromosome_20_variant_count', 'chromosome_20_variant_ratio',
        'read_depth_mean', 'read_depth_median', 'read_depth_std',
        'read_depth_quartile1', 'read_depth_quartile3', 'pl_mean',
        'pl_variance', '1', '10', '11', '12', '13', '14', '15', '16', '17',
        '18', '19', '2', '20', '3', '4', '5', '6', '7', '8', '9', 'vegetation'
    ]

    year_columns = ['2015', '2016', '2017', '2019', '2020', '2021', '2022', '2023']

    df_melted = train.melt(
        id_vars=id_vars,
        value_vars=year_columns,
        var_name='year',
        value_name='target'
    )

    log_step('Filtering melted data')
    df_filtered = df_melted.dropna(subset=['target']).reset_index(drop=True)
    df_filtered['year'] = df_filtered['year'].astype(int)

    log_step('Loading and merging weather data')
    weather_features = pd.read_csv(WEATHER_DATA_PATH)
    train_with_weather = pd.merge(df_filtered, weather_features, on='year', how='left')

    log_step('Saving final training data')
    train_with_weather.to_csv(r'../data/train_file.csv', index=False)
