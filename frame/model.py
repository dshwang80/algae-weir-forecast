# model.py — 로그공간 ridge (stdlib 전용, Gauss-Jordan 역행렬)
import math
from core import lg


def ridge_fit(X, y, lam=1e-3):
    """X: n×p (리스트), y: n — 리지지 회귀 (입력 사전 표준화 가정, 절편 열 내부 추가)
    β = (Xa'Xa+λI)^{-1} Xa'y  — Xa=[X|1]: 표준화 X는 원점 중심, y 평균 편향은 절편이 흡수"""
    p = len(X[0])
    Xa = [r + [1.0] for r in X]
    A = [[sum(Xa[i][a] * Xa[i][b] for i in range(len(Xa))) for b in range(p + 1)] for a in range(p + 1)]
    for a in range(p + 1):
        A[a][a] += lam
    b = [sum(Xa[i][a] * y[i] for i in range(len(Xa))) for a in range(p + 1)]
    return gauss_solve(A, b)


def gauss_solve(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            M[piv][c] = 1e-12
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                for j in range(c, n + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def standardize_fit(X):
    p = len(X[0])
    mu = [sum(r[a] for r in X) / len(X) for a in range(p)]
    sd = [math.sqrt(sum((r[a] - mu[a]) ** 2 for r in X) / len(X)) or 1.0 for a in range(p)]
    return mu, sd


def standardize_apply(X, mu, sd):
    return [[(r[a] - mu[a]) / sd[a] for a in range(len(r))] for r in X]
