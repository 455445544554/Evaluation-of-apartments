import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
import catboost
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor


# Отображение таблицы
data = pd.read_csv('b_data.csv')
display(data)

# Удаление городов не из московской области
spisok = ["Верея", "Высоковск", "Дрезна", "Талдом", "Руза", "Краснозаводск",
  "Пересвет", "Яхрома", "Голицыно", "Волоколамск", "Рошаль", "Кубинка",
  "Куровское", "Пущино", "Электроугли", "Черноголовка", "Хотьково",
  "Звенигород", "Бронницы", "Электрогорск", "Зарайск", "Старая Купавна",
  "Озёры", "Лосино-Петровский", "Красноармейск", "Ликино-Дулёво", "Можайск",
  "Луховицы", "Дедовск", "Апрелевка", "Шатура", "Истра", "Протвино",
  "Краснознаменск", "Кашира", "Котельники", "Солнечногорск", "Дзержинский",
  "Лыткарино", "Фрязино", "Павловский Посад", "Наро-Фоминск", "Ступино",
  "Дмитров", "Чехов", "Егорьевск", "Дубна", "Видное", "Клин", "Ивантеевка",
  "Лобня", "Воскресенск", "Сергиев Посад", "Ногинск", "Жуковский", "Пушкино",
  "Реутов", "Долгопрудный", "Орехово-Зуево", "Раменское", "Щёлково",
  "Серпухов", "Одинцово", "Домодедово", "Коломна", "Электросталь",
  "Красногорск", "Люберцы", "Королёв", "Мытищи", "Химки", "Подольск", "Балашиха",
          "Москва"]

data = data[data['location'].isin(spisok)]

# Конвертирование из объектов в float
data['kitchen_meters'] = data['kitchen_meters'].astype(float)
data['year_of_construction'] = data['year_of_construction'].astype(float)

# Удаление дубликатов
data.drop_duplicates()

# Преобразование строковых значений в числа, если возможно
def try_convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value

spisok = ["url", "location", "floor", "floors_count", "rooms_count",
          "total_meters", "price_feb",	"price_apr",	"price_may",
          "district", "street", "residential_complex",
          "living_meters",	"kitchen_meters",	"year_of_construction"]

for i in spisok:
  data[i] = data[i].apply(try_convert_to_int)

# Замена чисел на NaN
def replace_numbers_with_nan(x):
    if isinstance(x, (int, float)):
        return np.nan
    else:
        return x

spisok = ["url", "location", "district", "street",
          "residential_complex"]

for i in spisok:
  data[i] = data[i].apply(replace_numbers_with_nan)

# Подсчёт отрицательных значений в столбцах
spisok = ["floor", "floors_count", "rooms_count",
          "total_meters", "price_feb",	"price_apr",
          "price_may", "living_meters",
          "kitchen_meters",	"year_of_construction"]

data1 = data

for i in spisok:
  # Преобразование столбца в числовой формат с пропуском значений, которые не могут быть преобразованы
  data1[i] = pd.to_numeric(data1[i], errors='coerce')

  # Подсчёт количества отрицательных значений в столбце
  negative_count = data1[data1[i] < 0][i].count()

  print(f"Количество отрицательных значений в столбце {i}:", negative_count)

# Замена отрицательных значений на NaN
def replace_negative_with_nan(x):
    if isinstance(x, (int, float)) and x < 0:
        return np.nan
    else:
        return x

spisok = ["floor", "floors_count", "rooms_count", "total_meters",
          "price_feb",	"price_apr",	"price_may", "living_meters",
          "kitchen_meters",	"year_of_construction"]

for i in spisok:
  data[i] = data[i].apply(replace_negative_with_nan)

# Подсчёт отрицательных значений в столбцах
spisok = ["floor", "floors_count", "rooms_count", "total_meters",
          "price_feb",	"price_apr",	"price_may", "living_meters",
          "kitchen_meters",	"year_of_construction"]

for i in spisok:
  # Преобразование столбца в числовой формат с пропуском значений, которые не могут быть преобразованы
  data[i] = pd.to_numeric(data[i], errors='coerce')

  # Подсчёт количества отрицательных значений в столбце
  negative_count = data[data[i] < 0][i].count()

  print(f"Количество отрицательных значений в столбце {i}:", negative_count)

# Подсчёт количества пустых значений в каждом столбце
null_count = data.isnull().sum()

# Вывод столбцов, в которых есть пустые значения
print("Пустые значения в датасете:")
print(null_count[null_count > 0])

data.shape

# Удаление строк с пустыми значениями в столбце street
data = data.dropna(subset=["street"])

display(data)

# Замена всех NaN на "Отсутствует", так как не все города имеют деление по районам, а также не все здания относятся к ЖК
spisok = ["district",	"residential_complex"]

for i in spisok:
  data[i] = data[i].fillna('Отсутствует')

print(data.isnull().sum())

data[['living_meters', 'kitchen_meters', 'year_of_construction']] = data[['living_meters', 'kitchen_meters', 'year_of_construction']].replace(0, pd.NA)

data = data.dropna()
display(data)

print(data.isnull().sum())

# Подсчёт количества пустых значений в каждом столбце
null_count = data.isnull().sum()

# Вывод столбцов, в которых есть пустые значения
print("Пустые значения в датасете:")
print(null_count[null_count > 0])

# Отображение диаграммы
data.plot(kind='scatter', x='price_may', y='total_meters')

# Настройка осей и заголовка диаграммы
plt.xlabel(f'Цена от {format(round(min(data.price_may.values), 0), ",")} до {format(round(max(data.price_may.values), 0), ",")}')
plt.ylabel('Площадь')
plt.title('Цена и площадь квартир')
plt.xticks([])
# Отображение диаграммы на графике
plt.show()

data.shape

# Конвертирование из объектов в float
data['kitchen_meters'] = data['kitchen_meters'].astype(float)
data['year_of_construction'] = data['year_of_construction'].astype(float)

data.describe()

# Удаление выбросов
def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    return data

# Применение функции для удаления выбросов
data = remove_outliers(data, "price_feb")
data = remove_outliers(data, "price_apr")
data = remove_outliers(data, "price_may")
data = remove_outliers(data, "year_of_construction")
data = remove_outliers(data, "total_meters")

display(data)

# Удаление выбросов
data = data.drop(data[data['kitchen_meters'] < 5].index)
data = data.drop(data[data['living_meters'] < 10].index)

data.describe()

data = remove_outliers(data, "price_may")# Отображение диаграммы
data.plot(kind='scatter', x='price_may', y='total_meters')

# Настройка осей и заголовка диаграммы
plt.xlabel(f'Цена от {format(round(min(data.price_may.values), 0), ",")} до {format(round(max(data.price_may.values), 0), ",")}')
plt.ylabel('Площадь')
plt.title('Цена и площадь квартир')
plt.xticks([])
# Отображение диаграммы на графике
plt.show()

spisok = ["district",	"street",	"residential_complex"]

for pere in spisok:
  # Подсчет количества уникальных значений в колонке location
  location_counts = data[pere].value_counts()

  location_counts_dict = location_counts.to_dict()

  # Определение уникальных значений из location_counts, которые встречаются только один раз
  unique_values_to_remove = location_counts[location_counts == 1].index

  # Удаление всех полей, где значение встречается только один раз
  data = data[~data[pere].isin(unique_values_to_remove)]

# Вывод данных после удаления
display(data)

# Удаление мусора из колонки район
pereb = ["Продается 2-х комнатная квартира в мк Пронина", "удобный выезд на Ярославское шоссе",
         "ЛУЧШАЯ кв-ра в е!Свежий ремонт3-комн. квартира", "озеро-пляж. Отличная экология. До ж/д станции Кубинка 7 минут транспортом. Удобный выезд на Минское",
         "расположенный в городе Балашиха всего в 8 км от МКАД", "планируется открытие в 2024 г.", "Лучшая просторная 1 к.кв. в е!1-комн. квартира",
         "Кв-ра в развитом е!1-комн. квартира"]

data = data[~data["district"].isin(pereb)]

# Вывод количества уникальных районов в таблице
cities = data['district'].value_counts().to_string()
print(cities)

# Вывод количества уникальных улиц в таблице
cities = data['street'].value_counts().to_string()
print(cities)


cities = data['street'].value_counts().head(17)  # Ограничение
labels = cities.index
values = cities.values

# Создание круговой диаграммы
plt.pie(values, labels=labels)

# Добавление заголовка
plt.title('Распределение самых частых улиц продажи')

# Вывод диаграммы
plt.show()

# Удаление строк, с total_meters равному 0
data = data[data['total_meters'] != 0]

display(data)

# Конвертирование из объектов в float
data['kitchen_meters'] = data['kitchen_meters'].astype(float)
data['year_of_construction'] = data['year_of_construction'].astype(float)

data.describe()

data.shape

# Создание пустой колонки price_per_m2
data['price_per_m2'] = None

# Перебираем строки датасета
for index, row in data.iterrows():
    # Расчет значения для колонки price_per_m2
    price_per_m2 = row['price_may'] / row['total_meters']
    # Заполнение значения в колонке price_per_m2
    data.loc[index, 'price_per_m2'] = price_per_m2

display(data)

# Перезапись индексов
data.reset_index(drop=True, inplace=True)

# Подсчет количества уникальных значений в колонке location
location_counts = data['location'].value_counts()

location_counts_dict = location_counts.to_dict()

print(location_counts_dict)

# Расчёт средней цены квадратного метра для каждого города
average_price_per_square_meter = data.groupby("location")['price_per_m2'].mean().sort_values(ascending=False).to_string()
print("Цена за квадратный метр по городам:\n")
print(average_price_per_square_meter)


average_price_per_square_meter = data.groupby("location")['price_per_m2'].mean()[:125]
labels = average_price_per_square_meter.index
values = average_price_per_square_meter.values

# Преобразование в pd.Series
cities_series = pd.Series(values, index=labels)

# Установка размеров полотна
plt.figure(figsize=(23, 4))

# Создание диаграммы
cities_series.plot(kind='bar')

# Добавление названий осей
plt.xlabel('Города')
plt.ylabel('Цена за квадратный метр')

# Вывод диаграммы
plt.show()

# Создаем словарь для хранения улиц по районам
street_by_district = {}

# Группируем датасет по районам и получаем уникальные улицы для каждого района
grouped = data.groupby('location')['street'].unique()

# Заполняем словарь улицами для каждого района
for district, street in grouped.items():
    street_by_district[district] = street.tolist()

# Выводим словарь на экран
print(street_by_district)

with open("street.json", "w") as file:
    json.dump(street_by_district, file)

# Создаем словарь для хранения улиц по районам
street_by_district = {}


# Группируем датасет по районам и получаем уникальные улицы для каждого района
grouped = data.groupby('street')['district'].unique()

# Заполняем словарь улицами для каждого района
for district, street in grouped.items():
    street_by_district[district] = street.tolist()

# Выводим словарь на экран
print(street_by_district)

with open("district.json", "w") as file:
    json.dump(street_by_district, file)


# Создаем словарь для хранения улиц по районам
street_by_district = {}

# Группируем датасет по районам и получаем уникальные улицы для каждого района
grouped = data.groupby('street')['residential_complex'].unique()

# Заполняем словарь улицами для каждого района
for district, street in grouped.items():
    street_by_district[district] = street.tolist()

# Выводим словарь на экран
print(street_by_district)

with open("residential_complex.json", "w") as file:
    json.dump(street_by_district, file)


# Сохранение изменений цен для отображения прогнозов
import numpy as np
from sklearn.linear_model import LinearRegression

# датасет Dynamics
df = {'location': [], 'feb_apr, %': [], 'apr_may, %': [], 'forecast, %': []}
df = pd.DataFrame(df)

unique_locations = list(data['location'].unique())

for n in unique_locations:
  masks = data['location'] == n
  result = data[masks]

  # количество строк
  num_rows = len(result)

  # Среднее изменение в руб
  total_prices = result[['price_feb', 'price_apr', 'price_may']].sum()
  average_prices = (total_prices / num_rows).astype(int)

  # Среднее изменение в %
  feb_apr = ((average_prices['price_apr'] - average_prices['price_feb']) / average_prices['price_feb']) * 100
  apr_may = ((average_prices['price_may'] - average_prices['price_apr']) / average_prices['price_apr']) * 100

  # Создаем матрицу X для линейной регрессии
  X = np.array([[1], [2], [3]])  # время изменений цен

  # Создаем матрицу Y для линейной регрессии
  Y = np.array([average_price_feb, average_price_apr, average_price_may])  # изменения цен

  # Создаем линейную регрессию
  lr_model = LinearRegression()
  lr_model.fit(X, Y)

  # Создаем матрицу для прогноза цены на следующий шаг
  X_pred = np.array([[4]])  # время следующего изменения цены

  # Проводим прогноз
  price_pred = int(lr_model.predict(X_pred)[0])

  # Изменение в процентах
  forecast = ((price_pred - average_price_may)/average_price_may) * 100

  # Новая строка
  df.loc[len(df.index)] = [n, round(feb_apr, 2), round(apr_may, 2), round(forecast, 2)]

display(df)

df.to_csv('price_dynamics.csv', index=False)


# установка новых значений категориальным данным для уменьшения занимаемоего места датафреймом, а также для быстродействия выполнения действий с данными
sp_2 = ["district",	"street", "residential_complex", "location"]
marks_list_1 = list(set(data[sp_2[0]].tolist()))
marks_list_2 = list(set(data[sp_2[1]].tolist()))
marks_list_3 = list(set(data[sp_2[2]].tolist()))
marks_list_4 = list(set(data[sp_2[3]].tolist()))

sp = [marks_list_1, marks_list_2, marks_list_3, marks_list_4] # столбцы со значениями, которые нужно нормализовать
categorical_values = {}
for column in sp_2:
    categorical_values[column] = {}
    unique_values = list(set(data[column].tolist()))
    for i, value in enumerate(unique_values):
        data[column].mask(data[column] == value, i, inplace=True)
        categorical_values[column][value] = i

for column, values_dict in categorical_values.items():
    print(f"Категория: {column}")
    for value, category in values_dict.items():
        print(f"Значение: {value}, Категория: {category}")
    print()


# Экспорт словарей
import json

for column, values_dict in categorical_values.items():
    with open(f"{column}_dictionary.json", "w") as file:
        json.dump(values_dict, file)


spisok = ["location",	"floor",	"floors_count",
          "rooms_count",	"total_meters",	"price_per_m2",
          "district",	"street", "residential_complex",
          "living_meters", "kitchen_meters", "year_of_construction"]

# Вычисление матрицы коррелиций
corr_matrix = data[spisok].corr()

# Отображение матрицы
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Матрица корреляций')
plt.show()


# Конвертирование из объектов в float
data['kitchen_meters'] = data['kitchen_meters'].astype(float)
data['year_of_construction'] = data['year_of_construction'].astype(float)
data['location'] = data['location'].astype(float)
data['district'] = data['district'].astype(float)
data['street'] = data['street'].astype(float)
data['residential_complex'] = data['residential_complex'].astype(float)
print(X_train.dtypes)


# Выбор признаков для обучения модели
spisok = ["location",	"floor",	"floors_count",
          "rooms_count",	"total_meters",

          "district",	"street", "residential_complex",
          "living_meters", "kitchen_meters", "year_of_construction"]

# Разделение данных на обучающую и тестовую выборки
X = data[spisok]
y = data['price_per_m2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Обучение модели регрессии
model = catboost.CatBoostRegressor(verbose=False)
model.fit(X_train, y_train)

# Параметры для случайного перебора
param_dist = {'n_estimators': sp_randint(100, 500)}
# Создание объекта RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=10, cv=5)
# Поиск лучших гиперпараметров
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Прогнозирование цены квадратного метра
y_pred = model.predict(X_test)

# Оценка точности модели тремя методами (Кросс-валидация, отложенная выборка, перекрёстная проверка по случайным блокам)
scores = cross_val_score(model, X_test, y_test, cv=5)
print("Метод кросс-валидации:",round(scores.mean(), 2))

accuracy = model.score(X_test, y_test)
print('Метод отложенной выборки:', round(accuracy, 2))

scores = cross_val_score(model, X, y, cv=(ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)))
print("Метод перекрестной проверки по случайным блокам:", round(scores.mean(), 2))

mae = mean_absolute_error(y_test, y_pred)
print("Метод MAE: ", round(mae, 2), sep="")


# Обучение модели регрессии
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Параметры для случайного перебора
param_dist = {'n_estimators': sp_randint(100, 500)}
# Создание объекта RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=10, cv=5)
# Поиск лучших гиперпараметров
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Прогнозирование цены квадратного метра
y_pred = model.predict(X_test)

# Оценка точности модели тремя методами (Кросс-валидация, отложенная выборка, перекрёстная проверка по случайным блокам)
scores = cross_val_score(model, X_test, y_test, cv=5)
print("Метод кросс-валидации:",round(scores.mean(), 2))

accuracy = model.score(X_test, y_test)
print('Метод отложенной выборки:', round(accuracy, 2))

scores = cross_val_score(model, X, y, cv=(ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)))
print("Метод перекрестной проверки по случайным блокам:", round(scores.mean(), 2))

mae = mean_absolute_error(y_test, y_pred)
print("Метод MAE: ", round(mae, 2), sep="")


# Обучение модели регрессии
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Параметры для случайного перебора
param_dist = {'n_estimators': sp_randint(100, 500)}
# Создание объекта RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_dist,
                                   n_iter=10, cv=5)
# Поиск лучших гиперпараметров
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Прогнозирование цены квадратного метра
y_pred = model.predict(X_test)

# Оценка точности модели тремя методами (Кросс-валидация, отложенная выборка, перекрёстная проверка по случайным блокам)
scores = cross_val_score(model, X_test, y_test, cv=5)
print("Метод кросс-валидации:",round(scores.mean(), 2))

accuracy = model.score(X_test, y_test)
print('Метод отложенной выборки:', round(accuracy, 2))

scores = cross_val_score(model, X, y, cv=(ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)))
print("Метод перекрестной проверки по случайным блокам:", round(scores.mean(), 2))

mae = mean_absolute_error(y_test, y_pred)
print("Метод MAE: ", round(mae, 2), sep="")


# Обучение модели регрессии
model = AdaBoostRegressor()
model.fit(X_train, y_train)

# # Параметры для случайного перебора
# param_dist = {'n_estimators': sp_randint(100, 500)}
# # Создание объекта RandomizedSearchCV
# random_search = RandomizedSearchCV(estimator=model,
#                                    param_distributions=param_dist,
#                                    n_iter=10, cv=5)
# # Поиск лучших гиперпараметров
# random_search.fit(X_train, y_train)
# best_model = random_search.best_estimator_

# Прогнозирование цены квадратного метра
y_pred = model.predict(X_test)

# Оценка точности модели тремя методами (Кросс-валидация, отложенная выборка, перекрёстная проверка по случайным блокам)
scores = cross_val_score(model, X_test, y_test, cv=5)
print("Метод кросс-валидации:",round(scores.mean(), 2))

accuracy = model.score(X_test, y_test)
print('Метод отложенной выборки:', round(accuracy, 2))

scores = cross_val_score(model, X, y, cv=(ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)))
print("Метод перекрестной проверки по случайным блокам:", round(scores.mean(), 2))

mae = mean_absolute_error(y_test, y_pred)
print("Метод MAE: ", round(mae, 2), sep="")


# Выбран алгорит CatBoost изза наименьшей погрешности
# Создание DataFrame с предсказанными и фактическими значениями
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Вывод таблицы с предсказанными значениями
display(predictions)

# Создание графика предсказанных и фактических значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Актуальные')
plt.ylabel('Прогноз')
plt.title('Актуальные и спрогнозированные данные')
plt.show()


# Сохранение обученной модели
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Сохранение обработанного датасета
data.to_csv('dataset.csv', index=False)