import argparse
from scripts.generate_train_dataset import get_train_dataset
from scripts.get_weather_data import get_weather_data
from scripts.lama_train import train_lama


def parse_args():
    parser = argparse.ArgumentParser(description='Soybean yield forecasting.')

    # Аргументы для получения погодных данных (по умолчанию координаты Воронежа и Курска и 2015-2017 и 2019-2023 года соответственно)
    parser.add_argument('--lat1', type=float, default=51.692479, help='Latitude for first location')
    parser.add_argument('--lon1', type=float, default=39.199195, help='Longitude for first location')
    parser.add_argument('--year_start1', type=int, default=2015, help='Start year for first location weather data')
    parser.add_argument('--year_end1', type=int, default=2017, help='End year for first location weather data')
    parser.add_argument('--lat2', type=float, default=51.734513, help='Latitude for second location')
    parser.add_argument('--lon2', type=float, default=36.155477, help='Longitude for second location')
    parser.add_argument('--year_start2', type=int, default=2019, help='Start year for second location weather data')
    parser.add_argument('--year_end2', type=int, default=2023, help='End year for second location weather data')

    # Аргументы для генерации датасета (пути к input и output файлам)
    parser.add_argument('--vcf_path', type=str, default='../data/genotypes.vcf', help='Path to VCF file')
    parser.add_argument('--csv_path', type=str, default='../data/parsed_vcf.csv', help='Output CSV file path')
    parser.add_argument('--h5_path', type=str, default='../data/genotypes.h5', help='Path to HDF5 file')
    parser.add_argument('--weather_path', type=str, default='../data/weather_season_data.csv',
                        help='Path to weather data file')
    parser.add_argument('--phenotypes_path', type=str, default='../data/phenotypes.tsv',
                        help='Path to phenotypes data file')
    parser.add_argument('--vegetation_path', type=str, default='../data/vegetation.tsv',
                        help='Path to vegetation data file')

    # Аргументы для обучения модели
    parser.add_argument('--train_path', type=str, default='../data/train_file.csv', help='Path to training data')
    parser.add_argument('--n_threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--n_folds', type=int, default=7, help='Number of folds for cross-validation')
    parser.add_argument('--random_state', type=int, default=52, help='Random state for reproducibility')
    parser.add_argument('--timeout', type=int, default=36000, help='Training timeout in seconds')
    parser.add_argument('--target_name', type=str, default='target', help='Name of target column')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Получаем погодныые данные
    get_weather_data(
        lat_1=args.lat1, lon_1=args.lon1, year_start_1=args.year_start1, year_end_1=args.year_end1,
        lat_2=args.lat2, lon_2=args.lon2, year_start_2=args.year_start2, year_end_2=args.year_end2
    )

    # Генерируем тренировочный датасет
    get_train_dataset(
        vcf_file_path=args.vcf_path, output_csv_file_path=args.csv_path, h5_file_path=args.h5_path,
        weather_data_path=args.weather_path, phenotypes_path=args.phenotypes_path,
        vegetation_path=args.vegetation_path
    )

    # Обучаем модель
    train_lama(
        train_path=args.train_path, n_threads=args.n_threads, n_folds=args.n_folds,
        random_state=args.random_state, timeout=args.timeout, target_name=args.target_name
    )
