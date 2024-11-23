import pandas as pd 
import numpy as np 

from scipy import stats
from scipy.stats import entropy

def extract_comprehensive_genomic_features(df):
    # Применение анализа к каждому сэмплу
    comprehensive_features = df.groupby('Sample').apply(advanced_sample_analysis).reset_index()
    
    return comprehensive_features

def advanced_sample_analysis(sample_data):
        features = {}
        
        # Детальный генотипический анализ
        features['total_variants'] = len(sample_data)
        for h_c in [
            '0/0', '0/1', '1/1', './.', '1/2', '0/2', '2/2', '1/3', '2/3', '0/3', '3/3'
        ]:
            features[f'{h_c}_homozygous_count'] = sum(sample_data['Genotype'] == h_c)

        # Хромосомный профиль с детализацией
        chrom_variant_distribution = sample_data['Chromosome'].value_counts()
        for chrom, count in chrom_variant_distribution.items():
            features[f'chromosome_{chrom}_variant_count'] = count
            features[f'chromosome_{chrom}_variant_ratio'] = count / len(sample_data)

        # Позиционный анализ
        features['position_entropy'] = stats.entropy(np.histogram(sample_data['Position'], bins=20)[0])
        features['position_median'] = np.median(sample_data['Position'])
        features['position_variance'] = np.var(sample_data['Position'])
        features['position_range'] = sample_data['Position'].max() - sample_data['Position'].min()
        features['position_density'] = len(sample_data) / features['position_range']
                
        # Продвинутый аллельный анализ
        features['allele_diversity'] = len(set(sample_data['Reference'] + sample_data['Alternate']))
        features['ref_to_alt_ratio'] = (sample_data['Reference'] != sample_data['Alternate']).mean()
        
        ref_variants = sample_data['Reference'].value_counts()
        alt_variants = sample_data['Alternate'].value_counts()
        
        for base in ['A', 'T', 'G', 'C']:
            features[f'ref_{base}_count'] = ref_variants.get(base, 0)
            features[f'alt_{base}_count'] = alt_variants.get(base, 0)
        
        # Статистики глубины чтения с расширенной информацией
        reads = sample_data['Raw read depth']
        features['read_depth_mean'] = reads.mean()
        features['read_depth_median'] = reads.median()
        features['read_depth_std'] = reads.std()
        features['read_depth_quartile1'] = reads.quantile(0.25)
        features['read_depth_quartile3'] = reads.quantile(0.75)

        # Генетическая изменчивость
        pl_values = sample_data['List of Phred-scaled genotype likelihoods'].str.split(',', expand=True).astype(float)
        features['pl_mean'] = pl_values.mean().mean()
        features['pl_max'] = pl_values.max().max()
        features['pl_variance'] = pl_values.var().mean()
        
        # Сложный анализ переходов между вариантами
        transition_matrix = {
            'AG': sum((sample_data['Reference'] == 'A') & (sample_data['Alternate'] == 'G')),
            'GA': sum((sample_data['Reference'] == 'G') & (sample_data['Alternate'] == 'A')),
            'CT': sum((sample_data['Reference'] == 'C') & (sample_data['Alternate'] == 'T')),
            'TC': sum((sample_data['Reference'] == 'T') & (sample_data['Alternate'] == 'C'))
        }
        
        for transition, count in transition_matrix.items():
            features[f'transition_{transition}_count'] = count
        
        # Распределение вариантов
        variant_types = sample_data.apply(lambda row: f"{row['Reference']}>{row['Alternate']}", axis=1)
        variant_counts = variant_types.value_counts(normalize=True)
        for i, (var_type, freq) in enumerate(variant_counts.items(), 1):
            features[f'variant_type_{i}'] = freq
            if i > 10:
                break
        
        return pd.Series(features)
    