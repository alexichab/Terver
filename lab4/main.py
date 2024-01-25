import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Генерация данных с матожиданиями 1, 2, 3 и отклонением sigma
means = [1, 2, 3]
sigma = 1
data_2d_sigma_05 = []

for mean in means:
    data_2d_sigma_05.append(np.random.normal(mean, sigma, (100, 2)))

data_2d_sigma_05 = np.concatenate(data_2d_sigma_05, axis=0)

# Применение метода k-средних
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_2d_sigma_05)
labels = kmeans.predict(data_2d_sigma_05)

# Визуализация кластеров
plt.figure(figsize=(12, 8))

# Цвета для исходных данных
original_colors = ['red', 'green', 'blue']

# Отображение исходных данных с их цветами
for i in range(3):
    sample_data = data_2d_sigma_05[i*100:(i+1)*100]
    plt.scatter(sample_data[:, 0], sample_data[:, 1], alpha=0.5, label=f'Исходная выборка {i+1}', color=original_colors[i])

# Цвета для кластеров
cluster_colors = ['orange', 'magenta', 'black']

# Отображение кластеризованных данных с цветами кластеров
for i in range(3):
    cluster_data = data_2d_sigma_05[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=1.0, edgecolor=cluster_colors[i], facecolor='none', s=100, label=f'Кластер {i+1}')

# Отображение центроидов кластеров
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label='Центроиды')

plt.title('Кластеры и исходные данные')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Определение точек, попавших не в свой кластер
misplaced_points = []
for i in range(3):
    original_sample = data_2d_sigma_05[i*100:(i+1)*100]
    misplaced = original_sample[labels[i*100:(i+1)*100] != i]
    misplaced_points.append(misplaced)

# Подсчёт количества ошибочно классифицированных точек в каждом кластере
misplaced_points_counts = [len(mp) for mp in misplaced_points]

# Вывод общего количества ошибочно классифицированных точек
print(f'Количество ошибочно классифицированных точек: {misplaced_points_counts}')

