# By aplying Y = W * x + b. It get a line that fit to the points.

# random dots
import numpy as np
num_puntos = 100
conjunto_puntos = []
for i in range(num_puntos):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    conjunto_puntos.append([x1, y1])
x_data = [v[0] for v in conjunto_puntos]
y_data = [v[1] for v in conjunto_puntos]

# show dots
import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'ro', label='Dots')
plt.legend()
plt.show()

# model TensorFlow
