import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import exp, cos

def chi(x):
    return 1 if x >= 0 else 0

def f(x):
    return 2 * cos(8 * x) + 2 * chi(3 * x - 3) - x**2

def dF(x):
    return exp(x) + 3 * x**2

def main():
    a, b = -8, 75
    integral_continuous, _ = quad(lambda x: f(x) * dF(x), a, b)

    jumps = []
    for xk in [-1, 2]:
        if a < xk < b:
            delta_F = 0
            delta_F += 2 * (chi(xk + 1) - chi(xk + 1 - 1e-9)) # для 2χ(x+1)
            delta_F += 2 * (chi(4 * xk - 8) - chi(4 * xk - 8 - 1e-9)) # для 2χ(4x−8)
            jumps.append(f(xk) * delta_F)

    integral_jumps = sum(jumps)

    total_integral = integral_continuous + integral_jumps

    print(f"Интеграл Лебега-Стильтьеса: {total_integral:.6f}")

def chi_graph(x):
    return np.where(x >= 0, 1, 0)

def plot_chi_functions():
    x = np.linspace(-6, 6, 1000)
    y_chi1 = 2 * chi_graph(x + 1)
    y_chi2 = 2 * chi_graph(4 * x - 8)

    _, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax in axs:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xticks(np.arange(-6, 6, 1))

    # Левый график — 2χ(x+1)
    axs[0].plot(x, y_chi1, label='2χ(x+1)', color='pink')
    axs[0].axvline(x=-1, color='black', linestyle='--', linewidth=1.5)
    axs[0].set_title("2χ(x+1)")
    axs[0].set_xlabel("x")
    axs[0].grid(True)
    axs[0].legend()

    # Правый график — 2χ(4x−8)
    axs[1].plot(x, y_chi2, label='2χ(4x−8)', color='orange')
    axs[1].axvline(x=2, color='black', linestyle='--', linewidth=1.5)
    axs[1].set_title("2χ(4x−8)")
    axs[1].set_xlabel("x")
    axs[1].grid(True)
    axs[1].legend()

    plt.suptitle("Графики функций 2χ(x+1), 2χ(4x−8)")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

if __name__ == "__main__":
    main()
    plot_chi_functions()
