"""Small rectangular Hungarian assignment without SciPy (rows <= columns)."""
import numpy as np


def linear_sum_assignment(cost):
    costs = np.asarray(cost).tolist()
    n, m = np.shape(cost)
    if n > m:
        columns, rows = linear_sum_assignment(np.asarray(cost).T)
        return rows, columns
    u, v = [0.] * (n + 1), [0.] * (m + 1)
    matched, previous = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        matched[0] = row
        column = 0
        distance, used = [float('inf')] * (m + 1), [False] * (m + 1)
        while True:
            used[column] = True
            current_row = matched[column]
            delta, next_column = float('inf'), 0
            for candidate in range(1, m + 1):
                if not used[candidate]:
                    reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                    if reduced < distance[candidate]:
                        distance[candidate], previous[candidate] = reduced, column
                    if distance[candidate] < delta:
                        delta, next_column = distance[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            parent = previous[column]
            matched[column] = matched[parent]
            column = parent
    assignment = [0] * n
    for column in range(1, m + 1):
        if matched[column]:
            assignment[matched[column] - 1] = column - 1
    return np.arange(n), np.array(assignment, dtype=int)
