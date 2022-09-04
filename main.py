from matplotlib import pyplot as plt
import csv
from typing import List, Tuple


Vector = List[float]  # список вещественных чисел


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


def calculate_sums2(x: Vector, y: Vector) -> Tuple[float, float, float, float]:
    """
    Вычисление s1, s2, s3, s4
    :param x: отсчёты аппроксимируемой функции по X
    :param y: отсчёты аппроксимируемой функции по Y
    :return: кортеж сумм: сумма(х), сумма(x*x), сумма(y), сумма(x*y)
    """
    n: int = len(x)  # количество отсчётов
    return (sum(x),                                # s1
            sum([x[i] * x[i] for i in range(n)]),  # s2
            sum(y),                                # s3
            sum([x[i] * y[i] for i in range(n)]))  # s4


def calculate_determinator2(a00: float, a01: float, a10: float, a11: float) -> float:
    """
    Вычисление определителя матрицы 2x2
    :param a00: 0й элемент 0й строки матрицы
    :param a01: 1й элемент 0й строки матрицы
    :param a10: 0й элемент 1й строки матрицы
    :param a11: 1й элемент 1й строки матрицы
    :return: определитель = a00 * a11 - a01 * a10
    """
    return a00 * a11 - a01 * a10


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
    # ширины столбцов
    col_widths: List[int] = [len(str(cols))] + \
                            [max(max(map(len, map(str, columns[i]))),
                                 max(map(len, headers[i])))
                             for i in range(cols)]
    col_separator: str = "|"  # разделитель столбцов от заголовков
    row_separator: str = "-"  # разделитель строк от заголовков
    header: str = build_row(["i"], headers, col_widths, col_separator)  # заголовок таблицы
    separator: str = row_separator * len(header)  # разделитель между заголовком и телом таблицы
    pos: int = header.find(col_separator)
    separator = separator[:pos] + "+" + separator[pos + 1:]
    body = ""  # тело таблицы
    for i in range(rows):
        body = body + build_row([str(i)], [str(column[i]) for column in columns], col_widths, col_separator) + "\n"
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


def main_linear(is_plots_required: bool) -> None:
    """
    Подпрограмма для линейной аппроксимации
    :param is_plots_required: нужно ли строить график
    """
    filename = "input.csv"  # имя файла с исходными данными для линейной аппроксимации

    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):
        raise ValueError

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    s1, s2, s3, s4 = calculate_sums2(x, y)  # промежуточные суммы

    # определители
    det: float = calculate_determinator2(s1, n,
                                         s2, s1)
    det_a0: float = calculate_determinator2(s1, s3,
                                            s2, s4)
    det_a1: float = calculate_determinator2(s3, n,
                                            s4, s1)

    # коэффициенты линейной аппроксимации
    a0: float = det_a0 / det
    a1: float = det_a1 / det

    f: Vector = [a1 * x[i] + a0 for i in range(n)]    # значения аппроксимирующей функции
    e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
    e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадратны ошибок

    SE = calculate_squared_error(e)  # сумма квадратов ошибок

    MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

    # вывод результатов

    print("Промежуточные суммы:")
    print(f"{s1=};\t{s2=};\t{s3=};\t{s4=}")

    print("Определители:")
    print(f"{det=};\t{det_a0=};\t{det_a1}")

    print("Коэффициенты линейной аппроксимирующей функции:")
    print(f"{a0=};\t{a1=}")


    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", "f=a0+a1*x", "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"{SE=}")
    print("Максимальная абсолютная ошибка:")
    print(f"{MAE=}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        title: str = f"Линейная аппроксимация f(x)={a1:3f}*x" + \
                     f"{'+' if a0 > 0 else ''}{a0:3f}"
        plt.title(title)

        plt.show()


def calculate_sums3(x: Vector, y: Vector) -> Tuple[float, float, float, float,
                                                   float, float, float]:
    """
    Вычисление s1, s2, s3, s4, s5, s6, s7
    :param x: отсчёты аппроксимируемой функции по X
    :param y: отсчёты аппроксимируемой функции по Y
    :return: кортеж сумм: сумма(х), сумма(x*x), сумма(x*x*x), сумма(x*x*x*x), сумма(y), сумма(x*y), сумма(x*x*y)
    """
    s1, s2, s3, s4, s5, s6, s7 = 0, 0, 0, 0, 0, 0, 0
    for xi, yi in zip(x, y):
        s1 += xi
        s2 += xi * xi
        s3 += xi * xi * xi
        s4 += xi * xi * xi * xi
        s5 += yi
        s6 += xi * yi
        s7 += xi * xi * yi
    return s1, s2, s3, s4, s5, s6, s7


def calculate_determinator3(a00: float, a01: float, a02: float,
                            a10: float, a11: float, a12: float,
                            a20: float, a21: float, a22: float) -> float:
    """
    Вычисление определителя матрицы 3x3
    :param a00: 0й элемент 0й строки матрицы
    :param a01: 1й элемент 0й строки матрицы
    :param a02: 2й элемент 0й строки матрицы
    :param a10: 0й элемент 1й строки матрицы
    :param a11: 1й элемент 1й строки матрицы
    :param a12: 2й элемент 1й строки матрицы
    :param a20: 0й элемент 2й строки матрицы
    :param a21: 1й элемент 2й строки матрицы
    :param a22: 2й элемент 2й строки матрицы
    :return:
    """
    return a00 * calculate_determinator2(a11, a12, a21, a22) - \
           a10 * calculate_determinator2(a01, a02, a21, a22) + \
           a20 * calculate_determinator2(a01, a02, a11, a12)


def main_square(is_plots_required: bool) -> None:
    """
    Подпрограмма для квадратичной аппроксимации
    :param is_plots_required: нужно ли строить график
    """
    filename = "input.csv"  # имя файла с исходными данными для квадратичной аппроксимации

    print(f"Начало считывания входных значений из файла \"{filename}\"")

    x, y = read_inputs(filename)  # отсчёты аппроксимируемой функции

    n = len(x)  # количество отсчётов
    if n != len(y):
        raise ValueError

    print(f"Успешно считано {n} значений:")
    print("x: " + "  ".join(map(str, x)))
    print("y: " + "  ".join(map(str, y)))

    s1, s2, s3, s4, s5, s6, s7 = calculate_sums3(x, y)  # промежуточные суммы

    # определители
    det: float = calculate_determinator3(n,  s1, s2,
                                         s1, s2, s3,
                                         s2, s3, s4)
    det_a0: float = calculate_determinator3(s5, s1, s2,
                                            s6, s2, s3,
                                            s7, s3, s4)
    det_a1: float = calculate_determinator3(n, s5, s2,
                                            s1, s6, s3,
                                            s2, s7, s4)
    det_a2: float = calculate_determinator3(n,  s1, s5,
                                            s1, s2, s6,
                                            s2, s3, s7)

    # коэффициенты квадратичной аппроксимации
    a0: float = det_a0 / det
    a1: float = det_a1 / det
    a2: float = det_a2 / det

    f: Vector = [a2 * x[i] * x[i] + a1 * x[i] + a0 for i in range(n)]    # аппроксимирующая функция
    e: Vector = [f[i] - y[i] for i in range(n)]       # ошибки воспроизведения
    e_square: Vector = [e[i] ** 2 for i in range(n)]  # квадратны ошибок

    SE = calculate_squared_error(e)  # сумма квадратов ошибок

    MAE = calculate_max_absolute_error(e)  # максимальная абсолютная ошибка

    # вывод результатов

    print("Промежуточные суммы:")
    print(f"{s1=};\t{s2=};\t{s3=};\t{s4=};\t{s5=};\t{s6=};\t{s7=}")

    print("Определители:")
    print(f"{det=};\t{det_a0=};\t{det_a1=};\t{det_a2=}")

    print("Коэффициенты квадратичной аппроксимирующей функции:")
    print(f"{a0=};\t{a1=};\t{a2=}")


    print("Расчёт точности воспроизведения:")
    print(build_table([x, y, f, e, e_square],
                      ["x", "y", "f=a0+a1*x+a2*x*x", "e=f-y", "e^2"]))
    print("Мера отклонения:")
    print(f"{SE=}")
    print("Максимальная абсолютная ошибка:")
    print(f"{MAE=}")

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        title: str = f"Квадратичная аппроксимация f(x)={a2:3f}*x*x" + \
                     f"{'+' if a0 > 0 else ''}{a1:3f}*x" + \
                     f"{'+' if a0 > 0 else ''}{a0:3f}"
        plt.title(title)

        plt.show()


if __name__ == "__main__":
    is_plots_required = True  # нужно ли строить графики
    # main_linear(is_plots_required)
    # main_square(is_plots_required)
