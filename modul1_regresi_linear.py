# Import library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Membuat data sembarang
penjualan = np.array([6,5,5,4,4,3,2,2,2,1])
harga = np.array([16000,18000,27000,34000,50000,68000,65000,81000,85000,90000])

print("Data Penjualan:", penjualan)
print("Data Harga:", harga)

# Visualisasi awal
plt.scatter(penjualan, harga)
plt.xlabel("Penjualan")
plt.ylabel("Harga")
plt.title("Scatter Plot Data")
plt.show()

# Membuat model regresi
penjualan = penjualan.reshape(-1,1)
model = LinearRegression()
model.fit(penjualan, harga)

# Visualisasi hasil regresi
plt.scatter(penjualan, harga, color='red')
plt.plot(penjualan, model.predict(penjualan))
plt.title("Model Regresi Linear Sederhana")
plt.xlabel("Penjualan")
plt.ylabel("Harga")
plt.show()
