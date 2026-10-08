"""
Симплекс әдісі (минимизация) - қосарлы (dual) есеп арқылы.

Стандартты минимизация есебі:
    Minimize   Z = c1*x1 + ... + cn*xn
    Subject to A * x >= b,   x >= 0,   c >= 0

Алгоритм қадамдары (LibreTexts 4.3 бөлімі бойынша):
    1. Есепті жазу
    2. Шектеулер мен максат функциясынан матрица жасау (төменгі жол - максат функциясы)
    3. Матрицаны транспонирлеу (жолдар мен бағандарды ауыстыру)
    4. Транспонирленген матрицадан қосарлы (dual) максимизация есебін жазу
    5. Қосарлы есепті симплекс әдісімен шешу (4.2 бөліміндегідей)
    6. Бастапқы есептің шешімі - соңғы таблицаның төменгі жолында,
       slack айнымалылар бағандарында. Минимум Z = dual есептің максимумы.
"""


# ---------------------------------------------------------------
# 1) Максимизациялау симплекс әдісі (4.2 бөлімі)
# ---------------------------------------------------------------
def print_tableau(T, headers, basis, title=""):
    if title:
        print(f"\n{title}")
    print("Базис".ljust(8) + "".join(h.rjust(10) for h in headers) + "C".rjust(10))
    for i, row in enumerate(T[:-1]):
        print(basis[i].ljust(8) + "".join(f"{v:10.3f}" for v in row))
    print("-" * (8 + 10 * len(headers) + 10))
    print("Z".ljust(8) + "".join(f"{v:10.3f}" for v in T[-1]))


def simplex_max_tableau(c, A, b, main_names, slack_names, verbose=True):
    """Максимизация есебін шешіп, соңғы таблицаны қайтарады."""
    m, n = len(A), len(c)
    if any(bi < 0 for bi in b):
        raise ValueError("Dual есепте оң жақ бөлік (b) теріс болмауы керек.")

    # Бастапқы таблица: негізгі айнымалылар | slack | Z | C
    T = []
    for i in range(m):
        slack = [1.0 if j == i else 0.0 for j in range(m)]
        T.append([float(v) for v in A[i]] + slack + [0.0, float(b[i])])
    T.append([-float(v) for v in c] + [0.0] * m + [1.0, 0.0])

    headers = list(main_names) + list(slack_names) + ["Z"]
    basis = list(slack_names)

    if verbose:
        print_tableau(T, headers, basis, "Dual есептің бастапқы таблицасы:")

    it = 0
    while True:
        bottom = T[-1][:-2]
        min_val = min(bottom)
        if min_val >= -1e-12:          # теріс сан қалмады -> дайын
            break
        col = bottom.index(min_val)    # pivot бағаны

        row, best = None, None         # pivot жолы (ең кіші қатынас)
        for i in range(m):
            if T[i][col] > 1e-12:
                q = T[i][-1] / T[i][col]
                if best is None or q < best:
                    best, row = q, i
        if row is None:
            raise ValueError("Есептің шешімі шектелмеген (unbounded).")

        it += 1
        if verbose:
            print(f"\n=== Итерация {it} ===")
            print(f"Енетін айнымалы: {headers[col]}, шығатын айнымалы: {basis[row]}, "
                  f"pivot = {T[row][col]:.3f}")

        pivot = T[row][col]
        T[row] = [v / pivot for v in T[row]]
        for i in range(m + 1):
            if i != row:
                f = T[i][col]
                T[i] = [T[i][j] - f * T[row][j] for j in range(len(T[i]))]
        basis[row] = headers[col]

        if verbose:
            print_tableau(T, headers, basis, "Pivoting-тен кейінгі таблица:")

    return T, headers, basis


# ---------------------------------------------------------------
# 2) Минимизация: транспонирлеу -> dual -> шешу -> жауап оқу
# ---------------------------------------------------------------
def transpose(M):
    return [list(col) for col in zip(*M)]


def simplex_min(c, A, b, verbose=True):
    """
    Minimize c·x  subject to A x >= b, x >= 0  (c >= 0).
    Қайтарады: (x айнымалылар сөздігі, минимум Z)
    """
    if any(ci < 0 for ci in c):
        raise ValueError("Бұл әдіс стандартты минимизация үшін: c >= 0 болуы керек.")

    m, n = len(A), len(c)   # m - шектеулер саны, n - айнымалылар саны

    # 2-қадам: матрица (шектеулер + төменгі жол)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(m)]
    M.append(list(map(float, c)) + [0.0])
    if verbose:
        print("Бастапқы матрица (төменгі жол - максат функциясы):")
        for row in M:
            print("  " + "  ".join(f"{v:8.2f}" for v in row))

    # 3-қадам: транспонирлеу
    Mt = transpose(M)
    if verbose:
        print("\nТранспонирленген матрица:")
        for row in Mt:
            print("  " + "  ".join(f"{v:8.2f}" for v in row))

    # 4-қадам: dual есеп: Maximize b·y, A^T y <= c
    A_dual = [row[:-1] for row in Mt[:-1]]   # n x m
    b_dual = [row[-1] for row in Mt[:-1]]    # c
    c_dual = Mt[-1][:-1]                     # b

    if verbose:
        print("\nDual (максимизация) есебі:")
        print("  Maximize Z = " + " + ".join(f"{c_dual[j]:g}y{j+1}" for j in range(m)))
        for i in range(n):
            lhs = " + ".join(f"{A_dual[i][j]:g}y{j+1}" for j in range(m))
            print(f"  {lhs} <= {b_dual[i]:g}")

    # 5-қадам: dual есепті шешу. Негізгі айнымалылар y, slack айнымалылар x.
    y_names = [f"y{j+1}" for j in range(m)]
    x_names = [f"x{i+1}" for i in range(n)]
    T, headers, basis = simplex_max_tableau(c_dual, A_dual, b_dual,
                                            y_names, x_names, verbose)

    # 6-қадам: жауап - төменгі жолдағы slack (x) бағандары
    bottom = T[-1]
    x_values = {x_names[i]: bottom[m + i] for i in range(n)}
    z_min = bottom[-1]
    return x_values, z_min


if __name__ == "__main__":
    # Мысал 4.3.1:
    #   Minimize Z = 12x1 + 16x2
    #   x1 + 2x2 >= 40
    #   x1 +  x2 >= 30
    c = [12, 16]
    A = [[1, 2],
         [1, 1]]
    b = [40, 30]

    x, z = simplex_min(c, A, b)

    print("\n=== НӘТИЖЕ ===")
    for name, value in x.items():
        print(f"{name} = {value:g}")
    print(f"Минималды Z = {z:g}")