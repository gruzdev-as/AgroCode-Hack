import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import timedelta
import time


def get_weather(lat, lon, start_date_str='2019-01-01', end_date_str='2023-12-30'):
    '''Получение погодных данных по API'''
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = 'https://archive-api.open-meteo.com/v1/archive'

    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date_str,
        'end_date': end_date_str,
        'hourly': [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
            "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl",
            "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
            "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
            "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
            "soil_temperature_28_to_100cm", "soil_temperature_100_to_255cm",
            "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
            "soil_moisture_100_to_255cm", "shortwave_radiation", "direct_radiation",
            "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
            "terrestrial_radiation"
        ]
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {'date': pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive='left'
    )}

    for idx, var_name in enumerate(params['hourly']):
        hourly_data[var_name] = hourly.Variables(idx).ValuesAsNumpy()

    return pd.DataFrame(hourly_data)


def calculate_growth_periods(planting_date):
    '''
    Рассчет даты для каждого периода роста на основе даты посадки.

    Args:
        planting_date (datetime): Дата посадки

    Returns:
        dict: Словарь, содержащий даты начала и окончания каждого периода роста
    '''
    periods = {
        'emergence': {
            'start': planting_date,
            'end': planting_date + timedelta(days=10)  # 7-10 дней
        },
        'early_vegetative': {
            'start': planting_date + timedelta(days=10),
            'end': planting_date + timedelta(days=25)  # 10-15 дней после появления
        },
        'branching': {
            'start': planting_date + timedelta(days=25),
            'end': planting_date + timedelta(days=40)  # Медленный период роста перед цветением
        },
        'flowering': {
            'start': planting_date + timedelta(days=40),
            'end': planting_date + timedelta(days=75)  # 25-35 дней
        },
        'pod_formation': {
            'start': planting_date + timedelta(days=75),
            'end': planting_date + timedelta(days=95)  # 15-25 дней
        },
        'seed_filling': {
            'start': planting_date + timedelta(days=95),
            'end': planting_date + timedelta(days=115)  # 15-25 дней
        },
        'ripening': {
            'start': planting_date + timedelta(days=115),
            'end': planting_date + timedelta(days=130)  # 10-15 дней
        }
    }
    return periods


def generate_soybean_features(hourly_df, year, planting_date=None):
    '''
    Создание характеристик для анализа роста сои на основе физиологических периодов роста

    Args:
        hourly_df (pd.DataFrame): Почасовые данные о погоде
        year (int): Год для анализа
        planting_date (datetime, optional): Дата посадки. Если нет, то оценки на основе региональных условий
    '''
    # Если дата посадки не указана, оценивается на основе температуры почвы.
    if planting_date is None:
        # Первая дата, когда температура почвы постоянно выше 10°C (благоприятно для прорастания)
        spring_data = hourly_df[
            (hourly_df.index.year == year) &
            (hourly_df.index.month.isin([4, 5, 6]))  # Месяцы весны - лета
        ]
        temp_mask = spring_data['soil_temperature_0_to_7cm'] > 10
        planting_date = spring_data[temp_mask].index[0].to_pydatetime()

    print(planting_date)
    # Рассчитать периоды роста
    periods = calculate_growth_periods(planting_date)

    # Рассчитать производные характеристики
    hourly_df['temp_vs_apparent_diff'] = hourly_df['temperature_2m'] - hourly_df['apparent_temperature']
    hourly_df['heat_index'] = hourly_df['temperature_2m'] * (1 + hourly_df['relative_humidity_2m'] / 100)
    hourly_df['precipitation_intensity'] = hourly_df['precipitation'] / (hourly_df['rain'] + 1e-3)
    hourly_df['wind_chill'] = 13.12 + 0.6215 * hourly_df['temperature_2m'] - 11.37 * (
        hourly_df['wind_speed_10m'] ** 0.16) + 0.3965 * hourly_df['temperature_2m'] * (hourly_df['wind_speed_10m'] ** 0.16)

    # Характеристики почвы
    hourly_df['soil_temp_avg'] = hourly_df[['soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm',
                                           'soil_temperature_28_to_100cm', 'soil_temperature_100_to_255cm']].mean(axis=1)
    hourly_df['soil_temp_gradient'] = hourly_df['soil_temperature_0_to_7cm'] - hourly_df['soil_temperature_100_to_255cm']
    hourly_df['soil_moisture_surface'] = hourly_df[['soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm']].mean(axis=1)
    hourly_df['soil_moisture_deep'] = hourly_df[[
        'soil_moisture_28_to_100cm', 'soil_moisture_100_to_255cm']].mean(axis=1)
    hourly_df['soil_moisture_diff'] = hourly_df['soil_moisture_surface'] - hourly_df['soil_moisture_deep']

    features = {}

    # Рассчет характеристик для каждого периода роста
    for period_name, period_dates in periods.items():
        period_data = hourly_df[
            (hourly_df.index >= period_dates['start']) &
            (hourly_df.index <= period_dates['end'])
        ]

        if not period_data.empty:
            features[period_name] = {
                f'temp_mean': period_data['temperature_2m'].mean(),
                f'temp_max': period_data['temperature_2m'].max(),
                f'temp_min': period_data['temperature_2m'].min(),
                f'soil_temp_mean': period_data['soil_temp_avg'].mean(),
                f'soil_moisture_mean': period_data['soil_moisture_surface'].mean(),
                f'precipitation_sum': period_data['precipitation'].sum(),
                f'radiation_sum': period_data['shortwave_radiation'].sum(),
                f'vpd_mean': period_data['vapour_pressure_deficit'].mean(),
                f'etc_sum': period_data['et0_fao_evapotranspiration'].sum(),
                f'heat_stress_hours': len(period_data[period_data['temperature_2m'] > 30]),
                f'optimal_temp_hours': len(period_data[
                    (period_data['temperature_2m'] >= 15) &
                    (period_data['temperature_2m'] <= 30)
                ]),
                f'soil_temp_gradient_mean': period_data['soil_temp_gradient'].mean(),
                f'soil_moisture_diff_mean': period_data['soil_moisture_diff'].mean(),
                'period_length_days': (period_dates['end'] - period_dates['start']).days
            }

            # Добавить метрики, специфичные для определенного периода
            if period_name == 'emergence':
                features[period_name]['gdd_emergence'] = calculate_gdd(period_data, base_temp=10)
            elif period_name == 'flowering':
                night_hours = list(range(20, 24)) + list(range(0, 6))
                features[period_name]['night_temp_mean'] = period_data[
                    period_data.index.hour.isin(night_hours)
                ]['temperature_2m'].mean()
            elif period_name in ['pod_formation', 'seed_filling']:
                features[period_name]['drought_stress_hours'] = len(period_data[
                    (period_data['vapour_pressure_deficit'] > 2.0) &
                    (period_data['soil_moisture_surface'] < 0.2)
                ])

    # Рассчитать итоги сезона
    growing_season = hourly_df[
        (hourly_df.index >= periods['emergence']['start']) &
        (hourly_df.index <= periods['ripening']['end'])
    ]

    features['season_total'] = {
        'total_precipitation': growing_season['precipitation'].sum(),
        'total_radiation': growing_season['shortwave_radiation'].sum(),
        'total_heat_stress_days': len(growing_season[growing_season['temperature_2m'] > 30]) / 24,
        'total_optimal_temp_days': len(growing_season[
            (growing_season['temperature_2m'] >= 15) &
            (growing_season['temperature_2m'] <= 30)
        ]) / 24,
        'avg_soil_moisture': growing_season['soil_moisture_surface'].mean(),
        'avg_vpd': growing_season['vapour_pressure_deficit'].mean(),
        'total_etc': growing_season['et0_fao_evapotranspiration'].sum(),
        'season_length_days': (periods['ripening']['end'] - periods['emergence']['start']).days
    }

    return features


def calculate_gdd(weather_data, base_temp=10):
    """Рассчитать количество дней роста"""
    daily_temp = weather_data['temperature_2m'].resample('D').agg(['max', 'min'])
    daily_temp['avg'] = (daily_temp['max'] + daily_temp['min']) / 2
    gdd = (daily_temp['avg'] - base_temp).clip(lower=0).sum()
    return gdd


def analyze_soybean_growth(lat, lon, start_year=2019, end_year=2023, planting_dates=None):
    '''
    Полный анализ роста сои по годам

    Args:
        lat (float): Широта
        lon (float): Долгота
        start_year (int): Начальный год для анализа
        end_year (int): Заключительный год для анализа
        planting_dates (dict, optional): Словарь дат посадки по годам
    '''
    # Get weather data
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    hourly_df = get_weather(lat, lon, start_date, end_date)
    hourly_df.set_index('date', inplace=True)

    # Analyze each year
    yearly_features = {}
    for year in range(start_year, end_year + 1):
        planting_date = None if planting_dates is None else planting_dates.get(year)
        yearly_features[year] = generate_soybean_features(hourly_df, year, planting_date)

    return yearly_features


def print_growth_analysis(features, year):
    '''Анализ за определенный год'''
    print(f'\nSoybean Growth Analysis for {year}:')

    for period, metrics in features.items():
        if period != 'season_total':
            print(f'\n{period.replace('_', ' ').title()} Period:')
            for metric, value in metrics.items():
                if 'hours' in metric:
                    print(f"  {metric}: {value:.0f} hours")
                elif 'days' in metric:
                    print(f"  {metric}: {value:.1f} days")
                else:
                    print(f"  {metric}: {value:.2f}")

    print('\nSeason Totals:')
    for metric, value in features['season_total'].items():
        print(f'  {metric}: {value:.2f}')


def flatten_nested_dict(data):
    rows = []
    for year in data:
        row = {'year': year}

        for period in data[year]:
            if period == 'season_total':
                continue

            for metric, value in data[year][period].items():
                clean_metric = '_'.join(metric.split('_')[:-1]) if metric.split('_')[-1].isdigit() else metric
                column_name = f"{clean_metric}_{period}"
                row[column_name] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index('year', inplace=True)

    return df


def get_weather_data():
    lat = 51.692479
    lon = 39.199195

    final_voronezh = analyze_soybean_growth(lat, lon, 2015, 2017)
    voronezh_df = flatten_nested_dict(final_voronezh)

    time.sleep(3)

    lat = 51.734513
    lon = 36.155477

    final_kursk = analyze_soybean_growth(lat, lon, 2019, 2023)
    kursk_df = flatten_nested_dict(final_kursk)
    df = pd.concat([voronezh_df, kursk_df], axis=0)
    df.to_csv(r'..\data\weather_season_data.csv')
