import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/shic/OneDrive/Chen/S2_22_2170-88.csv", header = None)
df.columns = ["Time","Acceleration","Strain","Velocity"] 

a_axis = df[0:999]["Time"]
b_axis = df[0:999]["Acceleration"]
c_axis = df[0:999]["Strain"]
d_axis = df[0:999]["Velocity"]

plt.figure(figsize=(16,6))
plt.title("Time-Acceleration")
plt.plot(a_axis, b_axis, label="Time-Acceleration")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m^2/s)")

plt.grid(True)
plt.legend()

plt.show()