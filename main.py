from scripts.generate_train_dataset import get_train_dataset
from scripts.get_weather_data import get_weather_data
from scripts.lama_train import train_lama

# потом подтереть эти комменты
# добавить аргпарсер, в ламе параметры n_threads, n_folds, random_state, timeout, target_name по желанию, также сам путь к /data наверное
if __name__ == '__main__':
    # тут чисто функция которая получает погодные данные и сохраняет их в /data
    get_weather_data()
    # тут функция собирает статистики, генерит pca фичи и к этому всему добавляет погодную дату и сохраняет финальный трейн датасет
    get_train_dataset()
    # обучение модели на полученном датасете, выводятся метрики, фичимп, а также сохраняется полный отчет обучения
    train_lama()
