# Agrocode Hack Genetics

*MISIS Neychev Optimizer Team*

1) [**Груздев Александр**](https://github.com/gruzdev-as) - Captain (formal), ML
2) [**Рыжичкин Кирилл**](https://github.com/l1ghtsource) - ML
3) [**Литвинов Максим**](https://github.com/maksimlitvinov39kg) - ML 
4) [**Курочкин Вадим**](https://github.com/Vadimbuildercxx) - ML
5) [**Щелкунова Евгения**](https://github.com/jenyanya) - ML

Презентация: тык

## Кейс "Прогнозирование урожайности"

> Выход урожая — ключевая метрика для оценки успешности селекционного процесса, поэтому получение и отбор высокопродуктивных сортов — важная задача селекционера. Прогнозирование урожайности имеющихся в коллекции сортов позволит более эффективно выстраивать стратегию работы. На сегодняшний день имеются методы, основанные на статистике и машинном обучении, позволяющие использовать детальные данные о погоде за прошлые сезоны для прогнозирования выхода урожая, а также методы, использующие данные о геномах выращиваемых культур. Однако часто эти подходы используются независимо друг от друга. Совмещение этих подходов даст значительное увеличение точности предсказаний. Предложите, как может выглядеть подход, объединяющий детальные данные о почве и климате с геномными данными выращиваемых линий.

# Предложенное решение

## Блок-схема всего решения

тут умная схема сборки погодных данных, статистик, pca и подачи в модель

## Получение статистических признаков

Извлекаем статистические признаки из геномных данных, сгруппированных по образцам. Эта процедура позволяет получить количественные характеристики генетической информации, такие как распределение генотипов, профили хромосом, позиционные и аллельные особенности, глубину чтения и вероятность генотипов.

1. **Детальный генотипический анализ**:  
   Подсчитываются частоты различных генотипов (например, `0/0`, `0/1`, `1/1`, `./.`, и других). Это позволяет оценить распределение гомозиготных и гетерозиготных вариантов для каждого образца.

2. **Хромосомный профиль**:  
   Анализируется распределение генетических вариантов по хромосомам, включая абсолютное количество вариантов и их долю относительно общего числа вариантов в образце.

3. **Позиционный анализ**:  
   Исследуются пространственные характеристики расположения вариантов в геноме. Вычисляются такие показатели, как медиана, дисперсия, энтропия позиций, а также плотность распределения вариантов по геному.

4. **Аллельный анализ**:  
   Анализируется разнообразие аллелей. Подсчитываются частоты оснований (A, T, G, C) для референсных и альтернативных аллелей, а также вычисляется соотношение референсных и альтернативных аллелей.

5. **Анализ глубины чтения**:  
   Рассчитываются статистические показатели глубины чтения (среднее, медиана, стандартное отклонение, квартильные значения), что позволяет оценить качество секвенирования.

6. **Фред-вероятности (Phred-scaled likelihoods)**:  
   Извлекаются и анализируются вероятности генотипов в Phred-шкале. Вычисляются средние, максимальные значения и дисперсия вероятностей.

7. **Переходы между вариантами**:  
   Проводится анализ частот переходов между базами (например, A → G, C → T и т.д.), что может быть полезным для выявления характерных мутаций.

8. **Распределение типов вариантов**:  
   Генерируются типы замен (например, `A>G`, `C>T`), и их частоты нормализуются. Выводится распределение наиболее частых типов вариантов для каждого образца.

## Получение PCA признаков

## Получение погодных данных

## Финальная модель

```python
Final prediction for new objects (level 0) = 
	 0.31490 * (7 averaged models Lvl_0_Pipe_0_Mod_0_LinearL2) +
	 0.37158 * (7 averaged models Lvl_0_Pipe_1_Mod_2_CatBoost) +
	 0.31351 * (7 averaged models Lvl_0_Pipe_1_Mod_3_Tuned_CatBoost) 
```

Полный репорт об обучении и валидации: [report.html](logs/tabularAutoML_model_report_weather5/lama_interactive_report.html)

Важность фичей:

![imp](logs/__results___files/__results___116_1.png)
