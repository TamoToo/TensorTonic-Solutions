import numpy as np
from collections import deque

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    Returns: a tuple of two sorted coordinate lists: group and liberties.
    """
    n, m = len(board), len(board[0])
    color = board[row][col]
    liberties = set()
    queue = deque([(row, col)])
    visited = set([(row, col)])

    while queue:
        i, j = queue.popleft()
        for dir in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dir[0], j + dir[1]
                if 0 <= ni < n and 0 <= nj < m:
                    if board[ni][nj] == 0:
                        liberties.add((ni, nj))
                    elif not (ni, nj) in visited:
                        if board[ni][nj] == color:
                            visited.add((ni, nj))
                            queue.append((ni, nj))

    return sorted(visited), sorted(liberties)