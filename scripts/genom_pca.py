import allel 
import h5py
import numpy as np 

def parse_genotype_by_chromo(h5_file_path:str) -> dict:
    """Парсит H5 Файл и распределяет геномные данные 
       По соответствующим хромосомам

    Args:
        h5_file_path (str): Путь к Файлу

    Returns:
        dict: Словарь с информацией для каждой хромосомы (KZ... группируются вместе)
    """

    chrom_dict = {}

    with h5py.File(h5_file_path, 'r') as f:

        chrom_data = f['variants/CHROM']
        genotype_data = f['calldata/GT'][:]

        decoded_chrom = np.array(chrom_data).astype(str)

        for chromo in np.unique(decoded_chrom):
            
            if chromo.startswith('KZ'):
                continue

            chromo_seq = np.where(decoded_chrom == chromo)[0]
            chromo_seq_starts, chromo_seq_ends = chromo_seq[0], chromo_seq[-1]
            genotype_chromo = allel.GenotypeChunkedArray(genotype_data[chromo_seq_starts:chromo_seq_ends])
            chrom_dict[chromo] = genotype_chromo
        
        KZ_starts = np.argmax(np.char.startswith(decoded_chrom, 'KZ'))
        genotype_KZ =  allel.GenotypeChunkedArray(genotype_data[KZ_starts:])
        chrom_dict['KZ'] = genotype_KZ

    return chrom_dict

def get_pca_vectors(
        chrom_dict: dict,
        pca_parameters_dict:dict,
        plot_ld_flag:bool=True, 
        ld_prune_flag:bool=True,
        ld_prune_parameters_dict:dict=None, 
        ) -> dict:
    """ Получаем feature-вектор используя LD + PCA

    Args:
        chrom_dict (dict): Словарь для каждой хромосомы
        pca_parameters_dict (dict): Параметры для PCA
        plot_ld_flag (bool, optional): Печатать ли графики LD. Defaults to True.
        ld_prune_flag (bool, optional): Проводить или нет прунинг LD. Defaults to True.
        ld_prune_parameters_dict (dict, optional): Словарь параметров для прунинга. Defaults to None.

    Returns:
        dict: Словарь, где для каждой хромосомы полученный feature-вектор + PCA model
    """

    feature_dict = {}

    for chromo, gen in chrom_dict.items():
        
        ac = gen.count_alleles()[:]
        flt = (ac.max_allele() == 1) & (ac[:, :2].min(axis=1) > 1)
        gf = gen.compress(flt, axis=0)
        gn = gf.to_n_alt()

        if ld_prune_flag:
            gnu = ld_prune(gn, **ld_prune_parameters_dict)

        if plot_ld_flag:
            plot_pairwise_ld(gn[:1000], title='Pairwise LD')

            if ld_prune_flag:
                plot_pairwise_ld(gnu[:1000], title='Pairwise LD after LD pruning')

        if ld_prune_flag:
            coords, model = allel.pca(gnu, **pca_parameters_dict)
        else:
            coords, model = allel.pca(gn, **pca_parameters_dict)

        feature_dict[chromo] = {'feature_vector':coords, 'variance_model':model}
    
    return feature_dict


def plot_pairwise_ld(gn, title):
    """Рисует графики LD"""
    m = allel.rogers_huff_r(gn[:1000]) ** 2
    ax = allel.plot_pairwise_ld(m)
    ax.set_title(title)

def ld_prune(gn, size, step, threshold=.1, n_iter=1):
    """Проводит прунинг LD по заданым параметрам"""
    for i in range(n_iter):
        loc_unlinked = allel.locate_unlinked(gn, size=size, step=step, threshold=threshold)
        n = np.count_nonzero(loc_unlinked)
        n_remove = gn.shape[0] - n
        print('iteration', i+1, 'retaining', n, 'removing', n_remove, 'variants')
        gn = gn.compress(loc_unlinked, axis=0)
    return gn
