import numpy as np
import matplotlib.pyplot as plt

# Начальное приближение x0(t)
def x0(t):
    return t
    #return -7.5
    #return -4

# Система T(x)
y = []

def T(x, t):
    if 0 <= t <= 1/3:
        return (1/9) * x(3 * t) - 15/2 
    elif 1/3 < t < 2/3:
        t1, t2 = 1/3, 2/3
        y1, y2 = T(x, t1), T(x, t2)  # значения на границах y1 = -15/2, y2 = 0
        a = (y2 - y1) / (t2 - t1)  # коэффициент наклона a = 22,5
        b = y1 - a * t1  # свободный член b = -15
        y.append(y1)
        return a * t + b
    elif 2/3 <= t <= 1:
        return (1/9) * x(3 * t - 2)

# Итеративное применение оператора T
def find_fixed_point(x_init, epsilon=1e-6, max_iter=100):
    t_values = np.linspace(0, 1, 1000)  # Значения t
    x_current = x_init  # Текущая функция
    history = []  # Для хранения промежуточных итераций

    for i in range(max_iter):
        x_next_values = np.array([T(x_current, t) for t in t_values])  # Применяем T
        history.append((t_values, x_next_values.copy()))  # Сохраняем итерацию

        # Проверяем сходимость (норма разницы между итерациями < эпсилон)
        if i > 0:
            diff = np.max(np.abs(x_next_values - history[-2][1]))
            print(f"Итерация {i}: diff = {diff:.10f}")  # Выводим разницу
            if diff < epsilon:
                print(f"Сходимость достигнута на итерации {i}")
                break

        x_current = lambda t: x_next_values[np.argmin(np.abs(t_values - t))]  # Обновляем текущую функцию

    else:
        print("Достигнуто максимальное число итераций")

    return history  # Возвращаем историю итераций

def main():
    epsilon_values = [1e-1, 1e-2, 1e-3]

    for epsilon in epsilon_values:
        print(f"\nЗапуск для epsilon = {epsilon}")
        history = find_fixed_point(x0, epsilon=epsilon)
        print(y)
        plt.figure(figsize=(12, 6))

        # Левый подграфик: начальная и промежуточные итерации
        plt.subplot(1, 2, 1)
        
        # Добавляем начальную итерацию (нулевая итерация)
        t_values = history[0][0]
        x_init_values = np.array([x0(t) for t in t_values])
        plt.plot(t_values, x_init_values, label='Итерация 0', color='blue', linestyle='--', linewidth=2)

        # Добавляем промежуточные итерации
        for i, (t_values, x_values) in enumerate(history[1:]):
            plt.plot(t_values, x_values, label=f'Итерация {i + 1}')
        
        plt.axvline(x=1/3, color='gray', linestyle='--', label='t = 1/3')
        plt.axvline(x=2/3, color='black', linestyle='--', label='t = 2/3')
        plt.title(f'epsilon = {epsilon}\nНачальная и промежуточные итерации')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(-9, 1, 0.5))
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()

        # Правый подграфик: финальная итерация
        plt.subplot(1, 2, 2)
        t_values, x_final = history[-1]
        plt.plot(t_values, x_final, label='Финальная итерация', color='red', linewidth=2)
        plt.axvline(x=1/3, color='gray', linestyle='--', label='t = 1/3')
        plt.axvline(x=2/3, color='black', linestyle='--', label='t = 2/3')
        plt.title(f'epsilon = {epsilon}\nФинальная итерация')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(-9, 1, 0.5))
        plt.xlim(0, 1)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()