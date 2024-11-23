from scripts import get_weather_data, generate_train_dataset, lama_train

if __name__ == '__main__':
    get_weather_data()
    generate_train_dataset()
    lama_train()
