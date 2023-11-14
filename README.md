# **Базовые задачи**

## Задача 1:
Найдите среднее арифметическое и медиану списка чисел: [15, 20, 25, 30, 35, 40,45, 50, 55, 60].

```python
numbers = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

mean = sum(numbers) / len(numbers)
sorted_numbers = sorted(numbers)
median = sorted_numbers[len(sorted_numbers) // 2]

print("Среднее арифметическое:", mean)
print("Медиана:", median)
```

## Задача 2:
Вычислите дисперсию и стандартное отклонение для списка чисел: [3, 8, 12, 18, 25, 30].

```python
import numpy as np

numbers = [3, 8, 12, 18, 25, 30]

variance = np.var(numbers)
std_deviation = np.std(numbers)

print("Дисперсия:", variance)
print("Стандартное отклонение:", std_deviation)

numbers = [3, 8, 12, 18, 25, 30]

mean = sum(numbers) / len(numbers)
squared_diffs = [(x - mean) ** 2 for x in numbers]
variance = sum(squared_diffs) / len(numbers)
std_deviation = variance ** 0.5

print("Дисперсия:", variance)
print("Стандартное отклонение:", std_deviation)
```

## Задача 3:
Найдите корреляцию между двумя списками данных: [10, 15, 20, 25, 30] и [5, 10, 15, 20, 25].

```python
import numpy as np

data1 = [10, 15, 20, 25, 30]
data2 = [5, 10, 15, 20, 25]

correlation = np.corrcoef(data1, data2)[0, 1]

print("Корреляция:", correlation)

data1 = [10, 15, 20, 25, 30]
data2 = [5, 10, 15, 20, 25]

mean1 = sum(data1) / len(data1)
mean2 = sum(data2) / len(data2)

covariance = sum([(x - mean1) * (y - mean2) for x, y in zip(data1, data2)]) / len(data1)
std_deviation1 = (sum([(x - mean1) ** 2 for x in data1]) / len(data1)) ** 0.5
std_deviation2 = (sum([(x - mean2) ** 2 for x in data2]) / len(data2)) ** 0.5

correlation = covariance / (std_deviation1 * std_deviation2)

print("Корреляция:", correlation)
```

## Задача 4:
Рассчитайте вероятность броска симметричной монеты орлом 3 раза подряд.

```python
coin_prob = 0.5  # Вероятность выпадения орла или решки

probability_three_heads = coin_prob ** 3

print("Вероятность выпадения орла 3 раза подряд:", probability_three_heads)
```

## Задача 5:
Вычислите значение функции распределения для стандартного нормального распределения в точке x = 1.5.

```python
import scipy.stats as stats

x = 1.5
cdf_value = stats.norm.cdf(x)

print("Значение функции распределения:", cdf_value)

import math

def standard_normal_cdf(x):
    return (1 + math.erf(x / math.sqrt(2))) / 2

x = 1.5
cdf_value = standard_normal_cdf(x)

print("Значение функции распределения:", cdf_value)
```

## Задача 6:
Найдите обратное значение функции распределения для экспоненциального *распределения* с параметром λ = 0.2 в точке F(x) = 0.7.

```python
import scipy.stats as stats

x_inverse = stats.expon.ppf(0.7, scale=1/0.2)

print("Обратное значение функции распределения:", x_inverse)

def exponential_inverse_cdf(p, scale):
    return -scale * math.log(1 - p)

lambda_val = 0.2
p = 0.7

x_inverse = exponential_inverse_cdf(p, scale=1/lambda_val)

print("Обратное значение функции распределения:", x_inverse)
```

## Задача 7:
Смоделируйте выборку из нормального распределения с параметрами μ = 50 и σ = 10 размером 1000 элементов.

```python
mu = 50
sigma = 10
sample_size = 1000

sample = np.random.normal(mu, sigma, sample_size)

print("Первые 10 элементов выборки:", sample[:10])

import random

mu = 50
sigma = 10
sample_size = 1000

sample = [random.normalvariate(mu, sigma) for _ in range(sample_size)]

print("Первые 10 элементов выборки:", sample[:10])
```

## Задача 8:
Проверьте гипотезу, что среднее значение выборки [18.5, 20.1, 22.3, 19.8, 21.7] равно 20.

```python
from scipy.stats import ttest_1samp

sample = [18.5, 20.1, 22.3, 19.8, 21.7]
null_mean = 20

t_statistic, p_value = ttest_1samp(sample, null_mean)

print("Значение t-статистики:", t_statistic)
print("Значение p-значения:", p_value)

from scipy.stats import t

sample = [18.5, 20.1, 22.3, 19.8, 21.7]
null_mean = 20

sample_mean = sum(sample) / len(sample)
sample_std = (sum([(x - sample_mean) ** 2 for x in sample]) / (len(sample) - 1)) ** 0.5
sample_size = len(sample)
t_statistic = (sample_mean - null_mean) / (sample_std / (sample_size ** 0.5))
p_value = 2 * (1 - t.cdf(abs(t_statistic), df=sample_size - 1))

print("Значение t-статистики:", t_statistic)
print("Значение p-значения:", p_value)
```

## Задача 9:
Вычислите интервальный доверительный интервал для среднего значения выборки [120, 125, 130, 135, 140] с уровнем доверия 95%.

```python
from scipy.stats import t

sample = [120, 125, 130, 135, 140]
confidence_level = 0.95

sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
sample_size = len(sample)
margin_of_error = t.ppf((1 + confidence_level) / 2, df=sample_size - 1) * (sample_std / np.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print("Доверительный интервал:", confidence_interval)

sample = [120, 125, 130, 135, 140]
confidence_level = 0.95

sample_mean = sum(sample) / len(sample)
sample_std = (sum([(x - sample_mean) ** 2 for x in sample]) / (len(sample) - 1)) ** 0.5
sample_size = len(sample)
t_critical = t.ppf((1 + confidence_level) / 2, df=sample_size - 1)
margin_of_error = t_critical * (sample_std / (sample_size ** 0.5))

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print("Интервальный доверительный интервал:", confidence_interval)
```

## Задача 10:
Вычислите коэффициент детерминации (R^2) для линейной регрессии между переменными X и Y:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)

model = LinearRegression()
model.fit(X, Y)

R_squared = model.score(X, Y)

print("Коэффициент детерминации (R^2):", R_squared)

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

X_mean = sum(X) / len(X)
Y_mean = sum(Y) / len(Y)

numerator = sum([(x - X_mean) * (y - Y_mean) for x, y in zip(X, Y)])
denominator_X = sum([(x - X_mean) ** 2 for x in X])
denominator_Y = sum([(y - Y_mean) ** 2 for y in Y])

R_squared = (numerator ** 2) / (denominator_X * denominator_Y)

print("Коэффициент детерминации (R^2):", R_squared)
```


# **Задачи с графической интерпретацией**

## Задача 1:
Постройте гистограмму для выборки расходов клиентов магазина на покупку продуктов. Данные о расходах (в долларах): [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100].

```python
import matplotlib.pyplot as plt

expenses = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

plt.hist(expenses, bins=8, edgecolor='black')
plt.title("Распределение расходов на продукты")
plt.xlabel("Расходы ($)")
plt.ylabel("Частота")
plt.show()
```

## Задача 2:
Постройте ящик с усами для времени выполнения задач на сервере. Данные о времени выполнения (в миллисекундах): [120, 150, 180, 200, 220, 240, 260, 280, 300, 320, 350, 400, 450, 500, 600].

```python
data = [120, 150, 180, 200, 220, 240, 260, 280, 300, 320, 350, 400, 450, 500, 600]

plt.boxplot(data, vert=False, labels=["Время выполнения"])
plt.title("Ящик с усами времени выполнения задач")
plt.xlabel("Время (мс)")
plt.show()
```

## Задача 3:
Постройте график плотности вероятности для нормального распределения с параметрами μ = 50 и σ = 10.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu = 50
sigma = 10

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
pdf = stats.norm.pdf(x, mu, sigma)

plt.plot(x, pdf, color='blue')
plt.title("График плотности вероятности нормального распределения")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.show()
```

## Задача 4:
Постройте график распределения Пуассона с параметром λ = 5.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

lambda_val = 5
x = np.arange(0, 15)
pmf = stats.poisson.pmf(x, lambda_val)

plt.bar(x, pmf, color='green', alpha=0.7)
plt.title("График распределения Пуассона")
plt.xlabel("Значение")
plt.ylabel("Вероятность")
plt.show()
```

## Задача 5:
Постройте график экспоненциального распределения с параметром λ = 0.2.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

lambda_val = 0.2
x = np.linspace(0, 20, 100)
pdf = stats.expon.pdf(x, scale=1/lambda_val)

plt.plot(x, pdf, color='red')
plt.title("График плотности вероятности экспоненциального распределения")
plt.xlabel("Значение")
plt.ylabel("Плотность вероятности")
plt.show()
```

# **Модуль: "Numerical Linear Algebra"**

## Статистические методы

### 2.1.1. Доверительные интервалы:

Задача:
Вы анализируете результаты опроса, в котором 300 человек ответили на вопрос о своем уровне удовлетворенности продуктом. Средний уровень удовлетворенности составил 4.7 с стандартным отклонением 0.9. Постройте 95% доверительный интервал для среднего уровня удовлетворенности.

```python
import scipy.stats as stats
import numpy as np

confidence_level = 0.95
sample_mean = 4.7
sample_std = 0.9
sample_size = 300

margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * (sample_std / np.sqrt(sample_size))
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print("95% Доверительный интервал:", confidence_interval)
```

### 2.1.2. A/B тестирование:

Задача:
Вы проводите A/B тестирование для двух версий сайта с целью определить, какая из них приводит к большему количеству регистраций. Первая версия сайта показала 300 регистраций из 5000 посещений, вторая версия - 320 регистраций из 5200 посещений. Нужно ли вам считать, что различия в конверсии статистически значимыми на уровне значимости 0.05?

```python
import scipy.stats as stats

# Количество посещений и регистраций для двух версий
visits_A = 5000
registrations_A = 300
visits_B = 5200
registrations_B = 320

# Конверсии для каждой версии
conversion_A = registrations_A / visits_A
conversion_B = registrations_B / visits_B

# Вычисление статистической значимости с помощью z-теста
pooled_conversion = (registrations_A + registrations_B) / (visits_A + visits_B)
pooled_standard_error = ((pooled_conversion * (1 - pooled_conversion)) / visits_A + (pooled_conversion * (1 - pooled_conversion)) / visits_B) ** 0.5
z_score = (conversion_B - conversion_A) / pooled_standard_error
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

alpha = 0.05

if p_value < alpha:
    result = "статистически значимые"
else:
    result = "не статистически значимые"

print("P-значение:", p_value)
print("Результат A/B тестирования:", result)
```

### 2.1.3. Параметрические тесты:

Задача:
У вас есть две выборки данных о времени выполнения задач в миллисекундах для двух разных алгоритмов. Проверьте гипотезу, что среднее время выполнения задач с использованием первого алгоритма равно среднему времени выполнения задач со вторым алгоритмом на уровне значимости 0.05.

```python
import scipy.stats as stats

data_algorithm_1 = [120, 150, 180, 200, 220, 240, 260, 280, 300, 320]
data_algorithm_2 = [130, 155, 175, 210, 225, 245, 265, 285, 310, 330]

alpha = 0.05

t_statistic, p_value = stats.ttest_ind(data_algorithm_1, data_algorithm_2)

if p_value < alpha:
    result = "отвергаем"
else:
    result = "не отвергаем"

print("P-значение:", p_value)
print("Результат теста:", result)
```

### 2.1.4. Размер выборки, его связь с ошибкой:

Задача:
Вы хотите провести опрос среди клиентов интернет-магазина, чтобы оценить их уровень удовлетворенности. Сколько клиентов вам нужно опросить, чтобы при уровне доверия 95% и стандартном отклонении 0.8, ошибка оценки среднего уровня удовлетворенности составила не более 0.2?

```python
import scipy.stats as stats

confidence_level = 0.95
desired_margin_error = 0.2
sample_std = 0.8

z_score = stats.norm.ppf((1 + confidence_level) / 2)
sample_size = ((z_score * sample_std) / desired_margin_error) ** 2

print("Необходимый размер выборки:", round(sample_size))
```

### 2.2.1. Непараметрические тесты:

Задача:
У вас есть две выборки оценок студентов для двух разных методов обучения. Проверьте гипотезу, что оценки в обоих случаях имеют одинаковое распределение с использованием непараметрического критерия Манна-Уитни на уровне значимости 0.05.

```python
import scipy.stats as stats

data_method_1 = [85, 90, 88, 78, 92, 87, 89, 85, 95, 84]
data_method_2 = [75, 88, 85, 76, 82, 80, 88, 85, 90, 79]

alpha = 0.05

u_statistic, p_value = stats.mannwhitneyu(data_method_1, data_method_2)

if p_value < alpha:
    result = "отвергаем"
else:
    result = "не отвергаем"

print("P-значение:", p_value)
print("Результат теста:", result)
```

### 2.2.2. Бутстрап:

Задача:
Вы хотите оценить средний рост в популяции. Для этого у вас есть выборка из 50 человек. Постройте 95% доверительный интервал для среднего роста в популяции с использованием метода бутстрап.

```python
import numpy as np

# Ваша выборка (в сантиметрах)
sample_data = [165, 170, 172, 168, 175, 160, 180, 185, 175, 170,
               168, 167, 172, 174, 178, 182, 173, 169, 167, 169,
               171, 176, 177, 171, 174, 176, 172, 168, 173, 179,
               183, 168, 170, 169, 174, 175, 180, 176, 172, 167,
               166, 172, 171, 173, 176, 181, 185, 170, 178, 168]

# Количество бутстрап-подвыборок
num_bootstrap_samples = 10000

bootstrap_sample_means = []
for _ in range(num_bootstrap_samples):
    bootstrap_sample = np.random.choice(sample_data, size=len(sample_data), replace=True)
    bootstrap_sample_mean = np.mean(bootstrap_sample)
    bootstrap_sample_means.append(bootstrap_sample_mean)

confidence_level = 0.95
lower_percentile = (1 - confidence_level) / 2
upper_percentile = 1 - lower_percentile

lower_bound = np.percentile(bootstrap_sample_means, lower_percentile * 100)
upper_bound = np.percentile(bootstrap_sample_means, upper_percentile * 100)

print("95% Доверительный интервал:", (lower_bound, upper_bound))
```

### 2.2.3. Нелинейное преобразование данных:

Задача:
Вы исследуете зависимость между возрастом сотрудников и их производительностью. Постройте график рассеяния и определите, есть ли нелинейная зависимость между возрастом и производительностью. Если зависимость нелинейная, примените логарифмическое преобразование к производительности и постройте новый график рассеяния.

```python
import numpy as np
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)
ages = np.random.randint(20, 60, size=100)
productivity = 100 + 2 * ages + np.random.normal(0, 10, size=100)

# График рассеяния
plt.scatter(ages, productivity)
plt.title("Зависимость между возрастом и производительностью")
plt.xlabel("Возраст")
plt.ylabel("Производительность")
plt.show()

# Проверка на нелинейность
correlation = np.corrcoef(ages, productivity)[0, 1]

if abs(correlation) < 0.5:
    print("Зависимость нелинейная")

    # Логарифмическое преобразование
    log_productivity = np.log(productivity)

    # График рассеяния после преобразования
    plt.scatter(ages, log_productivity)
    plt.title("Зависимость между возрастом и логарифмом производительности")
    plt.xlabel("Возраст")
    plt.ylabel("Логарифм производительности")
    plt.show()
else:
    print("Зависимость линейная")
```

### 2.2.4. Множественная проверка:

Задача:
Вы тестируете эффективность новых лекарственных препаратов на 10 различных болезнях. Вам нужно применить поправку Бонферрони для управления ошибкой множественной проверки гипотез. Уровень значимости для каждой отдельной гипотезы составляет 0.05. Примените поправку Бонферрони и определите, какие гипотезы остаются статистически значимыми.

```python
import numpy as np
import scipy.stats as stats

# Генерация данных
np.random.seed(42)
p_values = np.random.rand(10) * 0.05

# Уровень значимости для множественной проверки
alpha_individual = 0.05
num_tests = len(p_values)
alpha_corrected = alpha_individual / num_tests

significant_hypotheses = []

for i, p_value in enumerate(p_values):
    if p_value < alpha_corrected:
        significant_hypotheses.append(i)

print("Список статистически значимых гипотез:", significant_hypotheses)
```

### 2.3.1. Матрица ковариации:

Задача:
У вас есть данные о росте (в сантиметрах) и весе (в килограммах) группы людей. Постройте матрицу ковариации для этих двух переменных.

```python
import numpy as np

# Данные о росте и весе (по столбцам)
data = np.array([
    [165, 60],
    [170, 65],
    [175, 70],
    [160, 55],
    [180, 75]
])

cov_matrix = np.cov(data, rowvar=False)
print("Матрица ковариации:")
print(cov_matrix)

import numpy as np

# Данные о росте и весе (по столбцам)
data = np.array([
    [165, 60],
    [170, 65],
    [175, 70],
    [160, 55],
    [180, 75]
])

# Вычисление средних
mean_values = np.mean(data, axis=0)

# Центрирование данных
mean_centered_data = data - mean_values

# Вычисление матрицы ковариации
num_samples = data.shape[0]
cov_matrix = np.dot(mean_centered_data.T, mean_centered_data) / (num_samples - 1)

print("Матрица ковариации:")
print(cov_matrix)
```

### 2.3.2. Применение метода главных компонент (PCA):

Задача:
У вас есть данные о росте (в сантиметрах), весе (в килограммах) и оценках (в баллах) группы людей. Примените метод главных компонент для снижения размерности данных до 2D и постройте новый график рассеяния.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Данные о росте, весе и оценках (по столбцам)
data = np.array([
    [165, 60, 85],
    [170, 65, 78],
    [175, 70, 92],
    [160, 55, 70],
    [180, 75, 88]
])

# Применение PCA
pca = PCA(n_components=2)
data_transformed = pca.fit_transform(data)

# График рассеяния
plt.scatter(data_transformed[:, 0], data_transformed[:, 1])
plt.title("Применение PCA")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Данные о росте, весе и оценках (по столбцам)
data = np.array([
    [165, 60, 85],
    [170, 65, 78],
    [175, 70, 92],
    [160, 55, 70],
    [180, 75, 88]
])

# Вычисление средних
mean_values = np.mean(data, axis=0)

# Центрирование данных
mean_centered_data = data - mean_values

# Вычисление матрицы ковариации
num_samples = data.shape[0]
cov_matrix = np.dot(mean_centered_data.T, mean_centered_data) / (num_samples - 1)

# Вычисление собственных значений и собственных векторов
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Сортировка собственных значений и векторов по убыванию
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Проекция данных на главные компоненты
data_transformed = np.dot(mean_centered_data, eigenvectors)

# График рассеяния
plt.scatter(data_transformed[:, 0], data_transformed[:, 1])
plt.title("Применение PCA")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.show()
```

### 2.3.3. SVD для расчёта PCA:

Задача:
У вас есть данные о росте (в сантиметрах) и доходе (в долларах) группы людей. Реализуйте метод главных компонент с использованием сингулярного разложения (SVD) и выведите главные компоненты.

```python
import numpy as np

# Данные о росте и доходе (по столбцам)
data = np.array([
    [165, 50000],
    [170, 55000],
    [175, 60000],
    [160, 45000],
    [180, 70000]
])

# Вычисление сингулярного разложения
mean_centered_data = data - np.mean(data, axis=0)
num_samples = data.shape[0]
cov_matrix = np.dot(mean_centered_data.T, mean_centered_data) / (num_samples - 1)
U, S, Vt = np.linalg.svd(cov_matrix, full_matrices=False)

# Главные компоненты
principal_components = Vt.T

print("Главные компоненты:")
print(principal_components)

import numpy as np

# Данные о росте и доходе (по столбцам)
data = np.array([
    [165, 50000],
    [170, 55000],
    [175, 60000],
    [160, 45000],
    [180, 70000]
])

# Вычисление средних
mean_values = np.mean(data, axis=0)

# Центрирование данных
mean_centered_data = data - mean_values

# Вычисление матрицы ковариации
num_samples = data.shape[0]
cov_matrix = np.dot(mean_centered_data.T, mean_centered_data) / (num_samples - 1)

# Вычисление собственных значений и векторов матрицы ковариации
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Сортировка собственных значений и векторов по убыванию
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Главные компоненты (собственные векторы)
principal_components = eigenvectors

print("Главные компоненты:")
print(principal_components)

### 2.4.1. Базовые понятия. Байесовский вывод:

Задача:
Представьте, что у вас есть монета, и вы хотите определить, является ли она справедливой (выпадение орла и решки равновероятно) или же она неправильная. Вы провели 10 подбрасываний и получили 7 раз орла и 3 раза решку. Известно, что у справедливой монеты вероятность выпадения орла или решки составляет 0.5. Используйте байесовский вывод, чтобы определить, какова вероятность того, что ваша монета справедливая.

# Априорная вероятность справедливой монеты
p_fair = 0.5

# Данные
num_flips = 10
num_heads = 7
num_tails = num_flips - num_heads

# Апостериорная вероятность справедливой монеты с учетом данных
p_fair_given_data = (p_fair * (0.5 ** num_heads) * (0.5 ** num_tails)) / (0.5 ** num_flips)

print("Вероятность справедливой монеты с учетом данных:", p_fair_given_data)
```

### 2.4.2. Статистическое моделирование. Априорная вероятность:

Задача:
Вы рассматриваете задачу классификации, где необходимо определить, является ли письмо спамом или нет. Вы создали бинарный классификатор и хотите использовать байесовский подход для принятия решения. Вам известно, что вероятность получения спам-письма составляет 0.2, а вероятность получения неспам-письма 0.8. Классификатор дает положительный результат для спама с вероятностью 0.9 и для неспама с вероятностью 0.1. Какова вероятность того, что письмо действительно является спамом, если классификатор дал положительный результат?

```python
# Априорные вероятности
p_spam = 0.2
p_not_spam = 0.8

# Вероятности классификации для спама и неспама
p_positive_given_spam = 0.9
p_positive_given_not_spam = 0.1

# Вычисление апостериорной вероятности спама при положительном результате классификации
p_spam_given_positive = (p_spam * p_positive_given_spam) / ((p_spam * p_positive_given_spam) + (p_not_spam * p_positive_given_not_spam))

print("Вероятность, что письмо спам, при положительном результате классификации:", p_spam_given_positive)
```

### 2.4.3. Выборка апостериорных распределений:

Задача:
Вы исследуете количество посетителей на вашем веб-сайте в течение недели. Вы собрали данные о числе посетителей каждый день и хотите оценить апостериорное распределение среднего числа посетителей. Ваша априорная вероятность для среднего числа посетителей составляет нормальное распределение с средним 100 и стандартным отклонением 10. Ваши наблюдения (посетители в течение недели) имеют нормальное распределение со средним 120 и стандартным отклонением 15. Оцените апостериорное распределение среднего числа посетителей.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Априорное распределение
prior_mean = 100
prior_std = 10

# Наблюдения
observations = np.array([110, 125, 130, 105, 140, 115, 120])

# Апостериорное распределение
posterior_std = 1 / np.sqrt(1/prior_std**2 + len(observations)/15**2)
posterior_mean = (prior_mean/prior_std**2 + np.sum(observations)/15**2) * posterior_std**2

# Вывод графика апостериорного распределения
x = np.linspace(80, 160, 1000)
prior_pdf = norm.pdf(x, prior_mean, prior_std)
posterior_pdf = norm.pdf(x, posterior_mean, posterior_std)

plt.plot(x, prior_pdf, label='Априорное')
plt.plot(x, posterior_pdf, label='Апостериорное')
plt.title("Априорное и апостериорное распределения среднего числа посетителей")
plt.xlabel("Среднее число посетителей")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.show()
```

### 2.4.4. Байесовская линейная регрессия:


Провести байесовский анализ данных, используя библиотеку Pyro, для оценки параметров линейной регрессии и создания визуализации постериорных распределений. Используйте предоставленный код и выполните следующие шаги:

1. Сгенерировать симулированные данные:
   - Создайте симулированные данные `X` и `Y` с линейной зависимостью и добавлением случайного шума.

2. Определить байесовскую модель регрессии:
   - Определите априорные распределения для параметров регрессии: наклон (`slope`), пересечение (`intercept`) и стандартное отклонение (`sigma`).
   - Опишите ожидаемое значение для зависимой переменной `mu`.
   - Задайте правдоподобие (распределение выборки) для наблюдений.

3. Выполнить байесовский вывод с использованием метода стохастической вариационной инференции (SVI):
   - Определите аппроксимирующие постериорные распределения для параметров.
   - Инициализируйте оптимизатор и SVI.
   - Запустите цикл вывода в течение `num_iterations` и отслеживайте функцию потерь.

4. Получить выборки из постериорного распределения с использованием Predictive:
   - Используйте `Predictive` для получения постериорных выборок параметров модели.

5. Вычислить средние значения параметров:
   - Вычислите средние значения для наклона (`slope`), пересечения (`intercept`) и стандартного отклонения (`sigma`) из полученных выборок.

6. Вывести оцененные параметры:
   - Вывести оцененные значения наклона, пересечения и стандартного отклонения.

7. Создать графики распределений параметров:
   - Построить три подграфика, отображающих постериорные распределения наклона, пересечения и стандартного отклонения.

8. Отобразить результаты:
   - Отобразить графики постериорных распределений параметров и вычисленные оценки параметров.

```python
!pip install pyro-ppl

#Import the necessary libraries
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns


# Generate some sample data
torch.manual_seed(0)
X = torch.linspace(0, 10, 100)
true_slope = 2
true_intercept = 1
Y = true_intercept + true_slope * X + torch.randn(100)

# Define the Bayesian regression model
def model(X, Y):
	# Priors for the parameters
	slope = pyro.sample("slope", dist.Normal(0, 10))
	intercept = pyro.sample("intercept", dist.Normal(0, 10))
	sigma = pyro.sample("sigma", dist.HalfNormal(1))

	# Expected value of the outcome
	mu = intercept + slope * X

	# Likelihood (sampling distribution) of the observations
	with pyro.plate("data", len(X)):
		pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)

# Run Bayesian inference using SVI (Stochastic Variational Inference)
def guide(X, Y):
	# Approximate posterior distributions for the parameters
	slope_loc = pyro.param("slope_loc", torch.tensor(0.0))
	slope_scale = pyro.param("slope_scale", torch.tensor(1.0),
							constraint=dist.constraints.positive)
	intercept_loc = pyro.param("intercept_loc", torch.tensor(0.0))
	intercept_scale = pyro.param("intercept_scale", torch.tensor(1.0),
								constraint=dist.constraints.positive)
	sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0),
						constraint=dist.constraints.positive)

	# Sample from the approximate posterior distributions
	slope = pyro.sample("slope", dist.Normal(slope_loc, slope_scale))
	intercept = pyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
	sigma = pyro.sample("sigma", dist.HalfNormal(sigma_loc))

# Initialize the SVI and optimizer
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

# Run the inference loop
num_iterations = 1000
for i in range(num_iterations):
	loss = svi.step(X, Y)
	if (i + 1) % 100 == 0:
		print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss}")

# Obtain posterior samples using Predictive
predictive = Predictive(model, guide=guide, num_samples=1000)
posterior = predictive(X, Y)

# Extract the parameter samples
slope_samples = posterior["slope"]
intercept_samples = posterior["intercept"]
sigma_samples = posterior["sigma"]

# Compute the posterior means
slope_mean = slope_samples.mean()
intercept_mean = intercept_samples.mean()
sigma_mean = sigma_samples.mean()

# Print the estimated parameters
print("Estimated Slope:", slope_mean.item())
print("Estimated Intercept:", intercept_mean.item())
print("Estimated Sigma:", sigma_mean.item())


# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot the posterior distribution of the slope
sns.kdeplot(slope_samples, shade=True, ax=axs[0])
axs[0].set_title("Posterior Distribution of Slope")
axs[0].set_xlabel("Slope")
axs[0].set_ylabel("Density")

# Plot the posterior distribution of the intercept
sns.kdeplot(intercept_samples, shade=True, ax=axs[1])
axs[1].set_title("Posterior Distribution of Intercept")
axs[1].set_xlabel("Intercept")
axs[1].set_ylabel("Density")

# Plot the posterior distribution of sigma
sns.kdeplot(sigma_samples, shade=True, ax=axs[2])
axs[2].set_title("Posterior Distribution of Sigma")
axs[2].set_xlabel("Sigma")
axs[2].set_ylabel("Density")

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
```

