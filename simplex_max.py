"""
Симплекс әдісі (максимизация) - қарапайым іске асыру.

Есеп (стандартты максимизация формасы):
    Maximize   Z = c1*x1 + c2*x2 + ... + cn*xn
    Subject to A * x <= b,   x >= 0,   b >= 0

Алгоритм қадамдары (LibreTexts 4.2 бөлімі бойынша):
    1. Есепті жазу
    2. Әр теңсіздікке slack айнымалы қосу
    3. Бастапқы симплекс-таблица құру (максат функциясы - төменгі жол)
    4. Төменгі жолдағы ең теріс элемент -> pivot бағаны
    5. Қатынастарды (quotients) есептеу, ең кішісі -> pivot жолы
    6. Pivoting (Гаусс-Жордан)
    7. Төменгі жолда теріс сан қалмағанша қайталау
    8. Жауапты оқу
"""


def print_tableau(T, headers, basis, title=""):
    """Таблицаны әдемі етіп шығару."""
    if title:
        print(f"\n{title}")
    print("Базис".ljust(8) + "".join(h.rjust(10) for h in headers) + "C".rjust(10))
    for i, row in enumerate(T[:-1]):
        print(basis[i].ljust(8) + "".join(f"{v:10.3f}" for v in row))
    print("-" * (8 + 10 * len(headers) + 10))
    print("Z".ljust(8) + "".join(f"{v:10.3f}" for v in T[-1]))


def simplex_max(c, A, b, verbose=True):
    m = len(A)       # шектеулер саны
    n = len(c)       # негізгі айнымалылар саны

    if any(bi < 0 for bi in b):
        raise ValueError("Бұл қарапайым нұсқа тек b >= 0 болғанда жұмыс істейді.")

    # --- 2-қадам: slack айнымалыларды қосу ---
    # --- 3-қадам: бастапқы таблицаны құру ---
    # Бағандар: x1..xn, y1..ym, Z, C
    T = []
    for i in range(m):
        slack = [1.0 if j == i else 0.0 for j in range(m)]
        T.append([float(v) for v in A[i]] + slack + [0.0, float(b[i])])
    # Төменгі жол: -c1 ... -cn, 0...0, 1, 0
    T.append([-float(v) for v in c] + [0.0] * m + [1.0, 0.0])

    headers = [f"x{j+1}" for j in range(n)] + [f"y{i+1}" for i in range(m)] + ["Z"]
    basis = [f"y{i+1}" for i in range(m)]

    if verbose:
        print_tableau(T, headers, basis, "Бастапқы симплекс-таблица:")

    iteration = 0
    while True:
        bottom = T[-1][:-2]  # Z және C бағандарынсыз

        # --- 7-қадам: тоқтау шарты ---
        min_val = min(bottom)
        if min_val >= -1e-12:
            break

        # --- 4-қадам: pivot бағаны (ең теріс элемент) ---
        col = bottom.index(min_val)

        # --- 5-қадам: қатынастар, ең кішісі -> pivot жолы ---
        row, best = None, None
        for i in range(m):
            if T[i][col] > 1e-12:            # нөл мен теріс элементтер ескерілмейді
                q = T[i][-1] / T[i][col]
                if best is None or q < best:
                    best, row = q, i
        if row is None:
            raise ValueError("Есептің шешімі шектелмеген (unbounded).")

        iteration += 1
        if verbose:
            print(f"\n=== Итерация {iteration} ===")
            print(f"Енетін айнымалы: {headers[col]} (бағандағы ең теріс мән {min_val:.3f})")
            print(f"Шығатын айнымалы: {basis[row]} (ең кіші қатынас {best:.3f})")
            print(f"Pivot элементі: {T[row][col]:.3f}")

        # --- 6-қадам: pivoting ---
        pivot = T[row][col]
        T[row] = [v / pivot for v in T[row]]                 # pivot = 1
        for i in range(m + 1):
            if i != row:
                factor = T[i][col]
                T[i] = [T[i][j] - factor * T[row][j] for j in range(len(T[i]))]
        basis[row] = headers[col]

        if verbose:
            print_tableau(T, headers, basis, "Pivoting-тен кейінгі таблица:")

    # --- 8-қадам: жауапты оқу ---
    solution = {h: 0.0 for h in headers[:-1]}
    for i, var in enumerate(basis):
        solution[var] = T[i][-1]
    z = T[-1][-1]
    return solution, z


if __name__ == "__main__":
    # Мысал 4.2.1 (Niki-дің екі жұмысы):
    #   Maximize Z = 40x1 + 30x2
    #   x1 +  x2 <= 12
    #  2x1 +  x2 <= 16
    c = [40, 30]
    A = [[1, 1],
         [2, 1]]
    b = [12, 16]

    sol, z = simplex_max(c, A, b)

    print("\n=== НӘТИЖЕ ===")
    for name, value in sol.items():
        print(f"{name} = {value:g}")
    print(f"Максималды Z = {z:g}")