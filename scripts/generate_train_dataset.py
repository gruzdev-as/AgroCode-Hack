import sys
import os
import pandas as pd
import allel
from scripts import utils, stat, genom_pca


def get_train_dataset(
    vcf_file_path=r'..\data\genotypes.vcf',
    output_csv_file_path=r'..\data\parsed_vcf.csv',
    h5_file_path=r'..\data\genotypes.h5',
    weather_data_path=r'..\data\weather_by_year_v2.csv'
):
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

    VCF_FILE_PATH = vcf_file_path,
    OUTPUT_CSV_FILE_PATH = output_csv_file_path,
    H5_FILE_PATH = h5_file_path
    WEATHER_DATA_PATH = weather_data_path

    utils.parse_vcf_2_csv(VCF_FILE_PATH, OUTPUT_CSV_FILE_PATH)

    df = pd.read_csv(r'..\data\parsed_vcf.csv')
    result = stat.extract_comprehensive_genomic_features(df)

    static_columns = [col for col in result.columns if result[col].nunique() == 1]
    new_result = result.drop(columns=static_columns, axis=1)

    allel.vcf_to_hdf5(
        input=VCF_FILE_PATH,
        output=H5_FILE_PATH,
        overwrite=True
    )

    LD_PRUNE_DICT = dict(size=100, step=20, threshold=.1, n_iter=1)
    PCA_PARAMETERS_DICT = dict(n_components=15, scaler='patterson')

    chromo_dict = genom_pca.parse_genotype_by_chromo(H5_FILE_PATH)
    feature_dict = genom_pca.get_pca_vectors(chromo_dict, PCA_PARAMETERS_DICT, False, True, LD_PRUNE_DICT)

    df_list = []
    for key, value in feature_dict.items():
        feature_vector = value['feature_vector']

        # Create a DataFrame for each key where each column corresponds to a sample
        sample_df = pd.DataFrame({
            f'Sample_{i+1}': [feature_vector[i]] for i in range(feature_vector.shape[0])
        })

        sample_df['Chroma'] = key

        df_list.append(sample_df)

    # Concatenate all DataFrames into a single DataFrame
    PCA_df = pd.concat(df_list, ignore_index=True)

    PCA_df = PCA_df.T
    PCA_df.columns = PCA_df.iloc[-1]
    PCA_df.to_csv(r'../data/20_chromas+KZ.csv')

    PCA_df = PCA_df.reset_index().drop('index', axis=1)

    new_result_w_pca20 = pd.concat([new_result, PCA_df], axis=1)

    ph = pd.read_csv(r'../data/phenotypes.tsv', sep='\t')
    vg = pd.read_csv(r'../data/vegetation.tsv', sep='\t')

    ph = ph.rename(columns={'sample': 'Sample'})
    vg = vg.rename(columns={'sample': 'Sample'})

    vg_w_ph = pd.merge(vg, ph, right_on='Sample', left_on='Sample')

    train = pd.merge(new_result_w_pca20, vg_w_ph, right_on='Sample', left_on='Sample')

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

    df_filtered = df_melted.dropna(subset=['target']).reset_index(drop=True)
    df_filtered['year'] = df_filtered['year'].astype(int)

    weather_features = pd.read_csv(WEATHER_DATA_PATH)
    train_with_weather = pd.merge(df_filtered, weather_features, on='year', how='left')

    train_with_weather.to_csv(r'../data/train_file.csv', index=False)
