import math

from matplotlib import pyplot as plt
import csv
from typing import List, Tuple, Callable

Vector = List[float]   # список вещественных чисел
Matrix = List[Vector]  # список векторов


def lin_space(start: float, stop: float, size: int) -> Vector:
    result: Vector = [0.0 for i in range(size)]
    step:   float  = (stop - start) / (size - 1)
    value:  float  = start
    for i in range(size):
        result[i] = value
        value    += step
    return result


def get_rows_and_cols_count(matrix: Matrix) -> Tuple[int, int]:
    """
    Получение размеров матрицы
    :param matrix: матрица
    :return:       кортеж [строки, столбцы]
    """
    return len(matrix), len(matrix[0])


def get_row(matrix: Matrix, row: int) -> Vector:
    rows, cols = get_rows_and_cols_count(matrix)
    result: Vector = list()
    for col in range(cols):
        result.append(matrix[row][col])
    return result


def get_col(matrix: Matrix, col: int) -> Vector:
    rows, cols = get_rows_and_cols_count(matrix)
    result: Vector = list()
    for row in range(rows):
        result.append(matrix[row][col])
    return result


def insert_row(matrix: Matrix, row: Vector, index: int) -> None:
    matrix.insert(index, row)


def add_row(matrix: Matrix, row: Vector) -> None:
    rows, cols = get_rows_and_cols_count(matrix)
    insert_row(matrix, row, rows)


def insert_col(matrix: Matrix, col: Vector, index: int) -> None:
    rows, cols = get_rows_and_cols_count(matrix)
    for row in range(rows):
        matrix[row].insert(index, col[row])


def add_col(matrix: Matrix, col: Vector) -> None:
    rows, cols = get_rows_and_cols_count(matrix)
    insert_col(matrix, col, cols)


def replace_col(matrix: Matrix, col: Vector, col_index: int) -> Matrix:
    """
    Замена столбца матрицы
    :param matrix:    исходная матрица
    :param col:       заменяющий столбец
    :param col_index: индекс столбца для замены в матрице
    :return:          матрица с замененным столбцом
    """
    rows, cols = get_rows_and_cols_count(matrix)
    result: Matrix = [[0 for j in range(cols)] for i in range(rows)]
    for j in range(cols):
        if j == col_index:
            for i in range(rows):
                result[i][j] = col[i]
        else:
            for i in range(rows):
                result[i][j] = matrix[i][j]
    return result


def copy_matrix(matrix: Matrix) -> Matrix:
    return [copy_vector(matrix[i]) for i in range(len(matrix))]


def copy_vector(vector: Vector) -> Vector:
    return [vector[i] for i in range(len(vector))]


def read_inputs(filename: str) -> Tuple[Vector, Vector]:
    """
    Чтение входных данных
    :param filename: имя файла с исходными данными
    :return: два списка, представляющих отсчёты аппроксимируемой функции по X и Y
    """
    # инициализация пустых списокв
    x: Vector = list()
    y: Vector = list()
    with open(filename, "r") as csv_file:  # открытие файла
        reader = csv.DictReader(csv_file)
        for row in reader:  # перебор строк
            # заполнение списков
            x.append(float(row["x"]))
            y.append(float(row["y"]))
    return x, y  # возврат результата


def calculate_squared_error(errors: Vector) -> float:
    """
    Вычисление суммы квадратов ошибки
    :param errors: ошибки
    :return: сумма квадратов ошибки
    """
    return sum([error ** 2 for error in errors])


def calculate_max_absolute_error(errors: Vector) -> float:
    """
    Вычисление максимальной абсолютной ошибки
    :param errors: ошибки
    :return: максимальная абсолютная ошибка
    """
    return max([abs(error) for error in errors])


def build_table(columns: List[Vector], headers: List[str]) -> str:
    """
    Построение таблицы с заданными колонками и заголовками
    :param columns: колонки значений
    :param headers: заголовки колонок
    :return: таблица в виде строки
    """
    rows: int = len(columns[0])  # количество строк таблицы
    cols: int = len(columns)  # количество колонок таблицы
    s_columns = [[str(column[i]) if type(column[i]) == int else f"{column[i]:.6}".strip() for column in columns] for i in range(rows)]
    # ширины столбцов
    col_widths: List[int] = [len(str(cols))] + \
                            [max(max(map(len, map(str, s_columns[i]))),
                                 len(headers[i]))
                             for i in range(cols)]
    col_separator: str = "|"  # разделитель столбцов от заголовков
    row_separator: str = "-"  # разделитель строк от заголовков
    header: str = build_row(["i"], headers, col_widths, col_separator)  # заголовок таблицы
    separator: str = row_separator * len(header)  # разделитель между заголовком и телом таблицы
    pos: int = header.find(col_separator)
    separator = separator[:pos] + "+" + separator[pos + 1:]
    body = ""  # тело таблицы
    for i in range(rows):
        body = body + build_row([str(i)], s_columns[i], col_widths, col_separator) + "\n"
    return header + "\n" + separator + "\n" + body


def build_row(lstrings: List[str], rstrings: List[str], col_widths: List[int], separator: str) -> str:
    """
    Построение строки таблицы
    :param lstrings: элементы до разделителя
    :param rstrings: элементы после разделителя
    :param col_widths: ширины столбцов
    :param separator: разделитель
    :return: строка таблицы
    """
    result: str = ""  # результат
    cnt: int = 0  # счётчик столбцов
    for lstring in lstrings:  # перебор строк до разделителя
        result = result + put_to_cell(lstring, col_widths[cnt])
        cnt += 1
    result = result + separator  # добавление разделителя
    for rstring in rstrings:  # перебор строк после разделителя
        result = result + put_to_cell(rstring, col_widths[cnt])
        cnt += 1
    return result


def put_to_cell(string: str, col_width: int) -> str:
    """
    Форматирование элемента таблицы
    :param string: элемент таблицы
    :param col_width: ширина столбца таблицы
    :return: элемент в нужном формате
    """
    return " " + " " * (col_width - len(string)) + string + " "


def build_header_function(k: int) -> str:
    """
    Получение строки с полиномом f(x)=a0+a1*x+...+ak*x^k
    :param k: степень полинома
    :return:  строка, представляющая полином
    """
    result = "f(x)="
    for i in range(k):
        result += f"+a{i}*x^{i}"
    return result


def build_title_function(coeffs: Vector) -> str:
    """
    Функция, аналогичная build_header_function, но коэффициенты полинома записываются числами
    :param coeffs: коэффициенты полинома
    :return:       строка, представляющая полином
    """
    result = "f(x)="
    for i, coeff in enumerate(coeffs):
        str_coeff = f"{coeff:.4}*x^{i}"
        if coeff > 0:
            str_coeff = "+" + str_coeff
        result += str_coeff
    return result


def calculate_sumsN(x: Vector, y: Vector, n: int) -> Vector:
    """
    Вычисление промежуточных сумм для алгоритма МНК
    :param x: отсчёты аппроксимируемой функции по X
    :param y: отсчёты аппроксимируемой функции по Y
    :param n: степень аппроксимирующего полинома
    :return:  набор сумм [sum(x), sum(x ** 2), ..., sum(x ** (2 * n)),
                          sum(x * y), sum(x ** 2 * y), ..., sum(x ** n * y)]
    """
    x_sums: Vector = [0.0 for i in range(2 * n)]
    y_sums: Vector = [0.0 for i in range(n + 1)]

    for xi, yi in zip(x, y):
        for i in range(2 * n):
            x_sums[i] += xi ** (i + 1)
        for i in range(n + 1):
            y_sums[i] += xi ** i * yi

    return x_sums + y_sums  # конкатенация массивов


def calculate_determinatorN(matrix: Matrix, result: float = 0.0) -> float:
    """
    Рекурсивное вычисление определителя путем разложения по первому столбцу
    :param matrix: квадртная матрица
    :param result: результат (для рекурсивных вызовов)
    :return:       определитель
    """
    rows: int = len(matrix)
    cols: int = len(matrix[0])
    if rows > 6:
        import numpy
        return numpy.linalg.det(matrix)

    if rows == 2:  # базовый случай - матрица 2х2
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    for row_index in range(rows):
        # "вырезание" подматрицы
        sub_matrix: Matrix = [[matrix[i][j] for j in range(1, cols)]
                                            for i in range(rows) if i != row_index]
        sign: int = (-1) ** row_index                         # знак для разложения
        sub_det: float = calculate_determinatorN(sub_matrix)  # определитель подматрицы
        result += sign * matrix[row_index][0] * sub_det       # разложение

    return result


def calculate_determinant(matrix: Matrix) -> float:
    rows, cols = get_rows_and_cols_count(matrix)
    extended = copy_matrix(matrix)
    for row in range(rows - 1):
        add_row(extended, get_row(matrix, row))
    result = 0.0
    for row in range(rows):
        main_mul = 1.0
        side_mul = 1.0
        for col in range(cols):
            main_mul *= extended[row + col][col]
            side_mul *= extended[(rows - 1) + row - col][col]
        result += main_mul - side_mul
    return result


def main_polynomial(filename: str, is_plots_required: bool, k: int) -> Vector:
    """
    Подпрограмма для полиномильной аппроксимации n-степени методом МНК
    :param filename:          имя файла с исходными данными для аппроксимации
    :param is_plots_required: нужно ли строить график
    :param k:                 степень аппроксимирующего полинома
    :returns Vector:          выходные значения аппроксимированной функции
    """
    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):       # если определитель близок к нулю
        raise ValueError  # выбрасывается исключение

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    sums = calculate_sumsN(x, y, k)  # промежуточные суммы

    left: Matrix = [[n if i + j == 0 else sums[i + j - 1] for j in range(k + 1)] for i in range(k + 1)]
    right: Vector = sums[len(sums) - (k + 1):]

    det: float = calculate_determinatorN(left)
    if abs(det) < 1e-9:
        raise ValueError

    coef_dets: Vector = [calculate_determinatorN(replace_col(left, right, i)) for i in range(k + 1)]
    coefs: Vector = [coef_det / det for coef_det in coef_dets]  # a0, a1, ..., ak

    f: Vector = [sum([coefs[j] * x[i] ** j for j in range(k + 1)]) for i in range(n)]    # аппроксимирующая функция
    e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
    e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадраты ошибок

    SE = calculate_squared_error(e)  # сумма квадратов ошибок

    MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

    # вывод результатов

    print("Промежуточные суммы:")
    print("; ".join(map(lambda x: f"s{x + 1}={sums[x]}", range(len(sums)))))

    print("Определители:")
    print(f"{det=}; " + "; ".join(map(lambda x: f"det_a{x}={coef_dets[x]}", range(len(coef_dets)))))

    print("Коэффициенты полиномиальной аппроксимирующей функции:")
    print("; ".join(map(lambda x: f"a{x}={coefs[x]}", range(len(coefs)))))

    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", build_header_function(k), "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"{SE=}")
    print("Максимальная абсолютная ошибка:")
    print(f"{MAE=}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        plt.grid()
        title: str = "Полиномиальная аппроксимация " + build_title_function(coefs)
        plt.title(title)

        plt.show()

    return f


def main_exponential(filename: str, is_plots_required: bool) -> Vector:
    """
    Подпрограмма для экспоненциальной аппроксимации методом МНК
    f(x) = a * exp(x) + b
    :param filename:          имя файла с исходными данными для аппроксимации
    :param is_plots_required: нужно ли строить график
    :returns Vector:          выходные значения аппроксимированной функции
    """

    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):       # если определитель близок к нулю
        raise ValueError  # выбрасывается исключение

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    s: Vector = [0.0, 0.0, 0.0, 0.0]  # промежуточные суммы

    for xi, yi in zip(x, y):       # перебираем значения x и y
        s[0] += math.exp(xi)       # sum(e^x)
        s[1] += math.exp(2 * xi)   # sum(e^2x)
        s[2] += yi                 # sum(y)
        s[3] += math.exp(xi) * yi  # sum(e^x * y)

    # определители
    det: float   = calculate_determinatorN([[s[1], s[0]],
                                            [s[0], n]])
    det_a: float = calculate_determinatorN([[s[3], s[0]],
                                            [s[2], n]])
    det_b: float = calculate_determinatorN([[s[1], s[3]],
                                            [s[0], s[2]]])

    if abs(det) < 1e-9:
        raise ValueError

    # коэффициенты экспоненциальной аппроксимации
    a: float = det_a / det
    b: float = det_b / det

    f: Vector = [a * math.exp(x[i]) + b for i in range(n)]    # аппроксимирующая функция
    e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
    e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадраты ошибок

    SE = calculate_squared_error(e)  # сумма квадратов ошибок

    MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

    # вывод результатов

    print("Промежуточные суммы:")
    print("; ".join(map(lambda x: f"s{x + 1}={s[x]}", range(len(s)))))

    print("Определители:")
    print(f"{det=}; {det_a=}; {det_b=}")

    print("Коэффициенты экспоненциальной аппроксимирующей функции:")
    print(f"{a=}; {b=}")

    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", "f=a*exp(x)+b", "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"{SE=}")
    print("Максимальная абсолютная ошибка:")
    print(f"{MAE=}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        plt.grid()
        title: str = "Экспоненциальная аппроксимация"
        plt.title(title)

        plt.show()

    return f


def main_exponential_with_k(filename: str, is_plots_required: bool) -> Vector:
    """
    Подпрограмма для экспоненциальной аппроксимации методом МНК
    f(x) = a * exp(k * x) + b
    :param filename:          имя файла с исходными данными для аппроксимации
    :param is_plots_required: нужно ли строить график
    :returns Vector:          выходные значения аппроксимированной функции
    """

    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):       # если определитель близок к нулю
        raise ValueError  # выбрасывается исключение

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    # поиск k с постепенным снижением области поиска и шага поиска
    k_max:    float = 10              # максимальная амплитуда поиска
    k_min:    float = 10e-6           # минимальная длина интервала поиска
    best_k:   float = 0.0             # середина интервала поиска
    best_SE:  float = float("inf")    # лучшая мера отколнения
    best_MAE: float = float("inf")    # лучшая абсолютная ошибка
    k_left:   float = best_k - k_max  # левая граница поиска
    k_right:  float = best_k + k_max  # правая граница поиска
    while abs(k_left - k_right) > k_min:  # пока интервал больше минимального значения
        k_values: Vector = lin_space(k_left, k_right, 21)  # откладываем от середины влево и вправо по 10 значений
        for k in k_values:  # перебираем значения k
            try:  # оборачиваем в try-except из-за возможных переполнений
                s: Vector = [0.0, 0.0, 0.0, 0.0]  # промежуточные суммы

                for xi, yi in zip(x, y):  # перебираем значения x и y
                    s[0] += math.exp(k * xi)       # sum(e^x)
                    s[1] += math.exp(2 * k * xi)   # sum(e^2x)
                    s[2] += yi                  # sum(y)
                    s[3] += math.exp(k * xi) * yi  # sum(e^x * y)

                # определители
                det: float   = calculate_determinatorN([[s[1], s[0]],
                                                        [s[0], n   ]])
                det_a: float = calculate_determinatorN([[s[3], s[0]],
                                                        [s[2], n   ]])
                det_b: float = calculate_determinatorN([[s[1], s[3]],
                                                        [s[0], s[2]]])

                if abs(det) < 1e-9:  # если определитель нулевой
                    continue         # берем следующее k

                # коэффициенты экспоненциальной аппроксимации
                a: float = det_a / det
                b: float = det_b / det

                f: Vector = [a * math.exp(k * x[i]) + b for i in range(n)]    # аппроксимирующая функция
                e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
                e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадраты ошибок

                SE = calculate_squared_error(e)  # сумма квадратов ошибок

                MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

                if SE < best_SE:   # если текущие результаты лучше наилучших
                    best_SE  = SE  # то запоминаем текущие результаты
                    best_MAE = MAE
                    best_k   = k

            except OverflowError:
                continue

        # после перебора всех k сужаем границы поиска
        k_left  = k_values[max(k_values.index(best_k) - 1, 0)]
        k_right = k_values[min(k_values.index(best_k) + 1, len(k_values) - 1)]

    # вывод результатов

    print("Промежуточные суммы:")
    print("; ".join(map(lambda x: f"s{x + 1}={s[x]}", range(len(s)))))

    print("Определители:")
    print(f"{det=}; {det_a=}; {det_b=}")

    print("Коэффициенты экспоненциальной аппроксимирующей функции:")
    print(f"{a=}; {b=}; k={best_k}")

    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", "f=a*exp(k*x)+b", "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"SE={best_SE}")
    print("Максимальная абсолютная ошибка:")
    print(f"MAE={best_MAE}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        plt.grid()
        title: str = "Экспоненциальная аппроксимация"
        plt.title(title)

        plt.show()

    return f


def main_logarithmic(filename: str, is_plots_required: bool) -> Vector:
    """
    Подпрограмма для логарифмической аппроксимации методом МНК
    f(x) = a * ln(x) + b
    :param filename:          имя файла с исходными данными для аппроксимации
    :param is_plots_required: нужно ли строить график
    :returns Vector:          выходные значения аппроксимированной функции
    """

    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):       # если определитель близок к нулю
        raise ValueError  # выбрасывается исключение

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    s: Vector = [0.0, 0.0, 0.0, 0.0]  # промежуточные суммы

    for xi, yi in zip(x, y):  # перебираем значения x и y
        s[0] += math.log(xi)       # sum(ln(x))
        s[1] += math.log(xi) ** 2  # sum(ln(x)^2)
        s[2] += yi                 # sum(y)
        s[3] += math.log(xi) * yi  # sum(ln(x) * y)

    # определители
    det: float   = calculate_determinatorN([[s[1], s[0]],
                                            [s[0],    n]])
    det_a: float = calculate_determinatorN([[s[3], s[0]],
                                            [s[2],    n]])
    det_b: float = calculate_determinatorN([[s[1], s[3]],
                                            [s[0], s[2]]])

    if abs(det) < 1e-9:
        raise ValueError

    # коэффициенты логарифмической аппроксимации
    a: float = det_a / det
    b: float = det_b / det

    f: Vector = [a * math.log(x[i]) + b for i in range(n)]    # аппроксимирующая функция
    e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
    e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадраты ошибок

    SE = calculate_squared_error(e)  # сумма квадратов ошибок

    MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

    # вывод результатов

    print("Промежуточные суммы:")
    print("; ".join(map(lambda x: f"s{x + 1}={s[x]}", range(len(s)))))

    print("Определители:")
    print(f"{det=}; {det_a=}; {det_b=}")

    print("Коэффициенты логарифмической аппроксимирующей функции:")
    print(f"{a=}; {b=}")

    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", "f=a*ln(x)+b", "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"{SE=}")
    print("Максимальная абсолютная ошибка:")
    print(f"{MAE=}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        plt.grid()
        title: str = "Логарифмическая аппроксимация"
        plt.title(title)

        plt.show()

    return f


def main_logarithmic_with_k(filename: str, is_plots_required: bool) -> Vector:
    """
    Подпрограмма для логарифмической аппроксимации методом МНК
    f(x) = a * ln(k + x) + b
    :param filename:          имя файла с исходными данными для аппроксимации
    :param is_plots_required: нужно ли строить график
    :returns Vector:          выходные значения аппроксимированной функции
    """

    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):       # если определитель близок к нулю
        raise ValueError  # выбрасывается исключение

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    # поиск k с постепенным снижением области поиска и шага поиска
    k_max:    float = 25              # максимальная амплитуда поиска
    k_min:    float = 10e-6           # минимальная длина интервала поиска
    best_k:   float = 25.0            # середина интервала поиска
    best_SE:  float = float("inf")    # лучшая мера отколнения
    best_MAE: float = float("inf")    # лучшая абсолютная ошибка
    k_left:   float = best_k - k_max  # левая граница поиска
    k_right:  float = best_k + k_max  # правая граница поиска
    while abs(k_left - k_right) > k_min:  # пока интервал больше минимального значения
        k_values: Vector = lin_space(k_left, k_right, 21)  # откладываем от середины влево и вправо по 10 значений
        for k in k_values:  # перебираем значения k
            try:  # оборачиваем в try-except из-за возможных переполнений
                s: Vector = [0.0, 0.0, 0.0, 0.0]  # промежуточные суммы

                for xi, yi in zip(x, y):  # перебираем значения x и y
                    s[0] += math.log(xi + k)       # sum(ln(x))
                    s[1] += math.log(xi + k) ** 2  # sum(ln(x)^2)
                    s[2] += yi                     # sum(y)
                    s[3] += math.log(xi + k) * yi  # sum(ln(x) * y)

                # определители
                det: float   = calculate_determinatorN([[s[1], s[0]],
                                                        [s[0],    n]])
                det_a: float = calculate_determinatorN([[s[3], s[0]],
                                                        [s[2],    n]])
                det_b: float = calculate_determinatorN([[s[1], s[3]],
                                                        [s[0], s[2]]])

                if abs(det) < 1e-9:  # если определитель нулевой
                    continue         # берем следующее k

                # коэффициенты экспоненциальной аппроксимации
                a: float = det_a / det
                b: float = det_b / det

                f: Vector = [a * math.log(k + x[i]) + b for i in range(n)]    # аппроксимирующая функция
                e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
                e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадраты ошибок

                SE = calculate_squared_error(e)  # сумма квадратов ошибок

                MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

                if SE < best_SE:   # если текущие результаты лучше наилучших
                    best_SE  = SE  # то запоминаем текущие результаты
                    best_MAE = MAE
                    best_k   = k

            except OverflowError:
                continue

        # после перебора всех k сужаем границы поиска
        k_left  = k_values[max(k_values.index(best_k) - 1, 0)]
        k_right = k_values[min(k_values.index(best_k) + 1, len(k_values) - 1)]

    # вывод результатов

    print("Промежуточные суммы:")
    print("; ".join(map(lambda x: f"s{x + 1}={s[x]}", range(len(s)))))

    print("Определители:")
    print(f"{det=}; {det_a=}; {det_b=}")

    print("Коэффициенты логарифмической аппроксимирующей функции:")
    print(f"{a=}; {b=}; k={best_k}")

    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", "f=a*ln(k+x)+b", "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"SE={best_SE}")
    print("Максимальная абсолютная ошибка:")
    print(f"MAE={best_MAE}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        plt.grid()
        title: str = "Логарифмическая аппроксимация"
        plt.title(title)

        plt.show()

    return f


PowerPair = Tuple[int, int]


class PowerPairs:
    @staticmethod
    def get_range(start: int, stop: int, min_sum: int, max_sum: int) -> List[PowerPair]:
        result = set()
        for first in range(start, stop):
            for second in range(start, stop):
                result.add((first, second))
        result = filter(lambda tup: min_sum <= sum(tup) <= max_sum, list(result))
        result = sorted(result, key=lambda tup: (sum(tup), max(tup), tup[1]))
        return list(result)

    @staticmethod
    def get_left_matrix(k: int) -> List[List[PowerPair]]:
        powers = PowerPairs.get_range(0, k + 1, 0, k)
        rows = len(powers)
        result = [powers]
        for i in range(1, rows):
            row = []
            for j in range(rows):
                row.append(PowerPairs.mul(powers[i], powers[j]))
            result.append(row)
        return result

    @staticmethod
    def mul(pair1: PowerPair, pair2: PowerPair) -> PowerPair:
        return pair1[0] + pair2[0], pair1[1] + pair2[1]

    @staticmethod
    def calculate_sum(x1: Vector, x2: Vector, cache_x: dict, powers: PowerPair, y: Vector = None, cache_y: dict = None):
        n = len(x1)
        if y is None:
            if powers in cache_x.keys():
                return cache_x[powers]
            if sum(powers) == 0:
                cache_x[powers] = n
                return n
            summa = 0.0
            for i in range(n):
                summa += x1[i] ** powers[0] * x2[i] ** powers[1]
            cache_x[powers] = summa
            return summa
        else:
            if powers in cache_y.keys():
                return cache_y[powers]
            summa = 0.0
            for i in range(n):
                summa += y[i] * x1[i] ** powers[0] * x2[i] ** powers[1]
            cache_y[powers] = summa
            return summa

    @staticmethod
    def calculate(k: int, x1: Vector, x2: Vector, y: Vector) -> Tuple[Matrix, Vector]:
        powers_matrix = PowerPairs.get_left_matrix(k)
        powers_vector = PowerPairs.get_range(0, k + 1, 0, k)
        result1 = []
        result2 = []
        cache_x = dict()
        cache_y = dict()
        rows = len(powers_vector)
        for i in range(rows):
            row = []
            for j in range(rows):
                row.append(PowerPairs.calculate_sum(x1, x2, cache_x, powers_matrix[i][j]))
            result1.append(row)
            result2.append(PowerPairs.calculate_sum(x1, x2, cache_x, powers_vector[i], y, cache_y))
        return result1, result2


def main_polynomial_two_inputs(k: int, function: Callable, x_min: float, x_max: float, steps: int, is_plots_required: bool) -> Vector:
    x1_linspace = lin_space(x_min, x_max, steps)
    x2_linspace = lin_space(x_min, x_max, steps)
    n = steps ** 2
    y:  Vector = [0.0 for i in range(n)]
    x1: Vector = [0.0 for i in range(n)]
    x2: Vector = [0.0 for i in range(n)]
    i = 0
    for x1_value in x1_linspace:
        for x2_value in x2_linspace:
            x1[i] = x1_value
            x2[i] = x2_value
            y[i] = function((x1_value, x2_value))
            i += 1
    left, right = PowerPairs.calculate(k, x1, x2, y)

    det: float = calculate_determinatorN(left)
    if abs(det) < 1e-9:
        raise ValueError
    powers_vector = PowerPairs.get_range(0, k + 1, 0, k)
    coefs_count = len(powers_vector)

    coef_dets: Vector = [calculate_determinatorN(replace_col(left, right, i)) for i in range(coefs_count)]
    coefs: Vector = [coef_det / det for coef_det in coef_dets]  # a0, a1, ..., ak

    f: Vector = [sum([coefs[j] * (x1[i] ** powers_vector[j][0]) * (x2[i] ** powers_vector[j][1]) for j in range(coefs_count)]) for i in range(n)]  # аппроксимирующая функция


    e: Vector = [f[i] - y[i] for i in range(n)]  # ошибки воспроизведения
    e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадраты ошибок

    SE = calculate_squared_error(e)  # сумма квадратов ошибок

    MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

    # вывод результатов

    print("Определители:")
    print(f"{det=}; " + "; ".join(map(lambda x: f"det_a{x}={coef_dets[x]}", range(len(coef_dets)))))

    print("Коэффициенты полиномиальной аппроксимирующей функции:")
    print("; ".join(map(lambda x: f"a{x}={coefs[x]}", range(len(coefs)))))

    print("Мера отклонения:")
    print(f"{SE=}")
    print("Максимальная абсолютная ошибка:")
    print(f"{MAE=}")

    if is_plots_required:  # формирование графика
        approximation = lambda args: sum(
            [args[2][j] * (args[0] ** args[3][j][0]) * (args[1] ** args[3][j][1]) for j in range(len(args[3]))])

        import numpy as np
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        x, y = np.linspace(x_min, x_max, 11), np.linspace(x_min, x_max, 11)
        X, Y = np.meshgrid(x, y)
        Z = approximation((X, Y, coefs, powers_vector))
        F = np.sin(np.pi / 2 * X) * np.cos(np.pi / 2 * Y)

        ax1.plot_surface(X, Y, Z, cmap='inferno')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        ax1.set_title("Аппроксимация")
        ax2.plot_surface(X, Y, F, cmap='inferno')
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Исходная функция")
        plt.show()

    return f


if __name__ == "__main__":
    filename = "input.csv"  # имя файла с исходными данными для аппроксимации
    is_plots_required = False  # нужно ли строить графики

    # main_polynomial(filename, is_plots_required, 1)
    # main_polynomial(filename, is_plots_required, 2)
    # main_polynomial(filename, is_plots_required, 3)
    # main_polynomial(filename, is_plots_required, 4)
    # main_polynomial(filename, is_plots_required, 5)

    # main_exponential(filename, is_plots_required)
    # main_exponential_with_k(filename, is_plots_required)

    # main_logarithmic(filename, is_plots_required)
    # main_logarithmic_with_k(filename, is_plots_required)


    function = lambda x: math.sin(math.pi / 2 * x[0]) * math.cos(math.pi / 2 * x[1])
    x_min = 0.0
    x_max = 1.0
    steps_count = 501
    main_polynomial_two_inputs(7, function, x_min, x_max, steps_count, is_plots_required)
