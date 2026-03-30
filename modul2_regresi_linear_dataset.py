# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("FuelConsumptionCo2.csv")

# Ambil kolom penting
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','CO2EMISSIONS']]

# Visualisasi
plt.scatter(cdf.FUELCONSUMPTION_CITY, cdf.CO2EMISSIONS)
plt.xlabel("Fuel Consumption City")
plt.ylabel("CO2 Emissions")
plt.show()

# Split data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Training model
model = LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

model.fit(train_x, train_y)

print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)

# Visualisasi hasil
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)
plt.plot(train_x, model.coef_[0][0]*train_x + model.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()
