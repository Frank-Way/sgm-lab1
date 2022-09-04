Ссылка на github репозиторий: 

Исходные данные для аппроксимации необходимо записать в файл "input.csv" и поместить в папку проекта. Формат файла:

    "x","y"
    <x1>,<y1>
    <x2>,<y2>
    ...
    <xn>,<yn>
Десятичный разделитель - ".", разделитель значений - "," (без пробелов!).

Выбор алгоритма линейной или квадратичной аппроксимации осуществляется путем раскомментирования одной из строчек вызова функции main_...:

    if __name__ == "__main__":
        is_plots_required = True  # нужно ли строить графики
        # перед запуском надо расскомментировать одну из следующих строк
        # main_linear(is_plots_required)  # линейная аппроксимация
        # main_square(is_plots_required)  # квадратичная аппроксимация

Основная часть скрипта реализована на "чистом Python" и не требует дополнительных библиотек. Для построения графиков используется библиотека matplotlib. При отсутствии возможности установки библиотеки необходимо закомментировать или удалить следующие строки:

    from matplotlib import pyplot as plt
...

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        title: str = f"Линейная аппроксимация f(x)={a1:3f}*x" + \
                     f"{'+' if a0 > 0 else ''}{a0:3f}"
        plt.title(title)

        plt.show()
...

    if is_plots_required:  # формирование графика
        plt.plot(x, y, "o", label="исходные данные")
        plt.plot(x, f, "-", label="аппроксимация")
        plt.legend(loc="best")
        title: str = f"Квадратичная аппроксимация f(x)={a2:3f}*x*x" + \
                     f"{'+' if a0 > 0 else ''}{a1:3f}*x" + \
                     f"{'+' if a0 > 0 else ''}{a0:3f}"
        plt.title(title)

        plt.show()

После этого скрипт может быть запущен "голым" интерпретатором (например, с помощью интерпретатора на сайте replit.com, который позволяет также загружать и создавать различные файлы в проекте, в том числе csv-файлы).