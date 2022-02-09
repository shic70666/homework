#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/shic/OneDrive/Chen/S2_22_2170-88.csv", header = None)
df.columns = ["Time","Acceleration","Strain","Velocity"] 

#%%
a_axis = df[:90]["Time"]
b_axis = df[:90]["Acceleration"]
c_axis = df[:90]["Strain"]
d_axis = df[:90]["Velocity"]

plt.figure(figsize=(16,6))
plt.title("Time-Acceleration")
plt.plot(a_axis, b_axis, label="Time-Acceleration")

plt.xlabel("Time (s)")
plt.ylabel(r"$Acceleration (m^2/s)$")

plt.grid(True)
plt.legend()

plt.show()

#%%
plt.figure(figsize=(16,12))
plt.subplot(211)
plt.title("Time-Acceleration")
plt.plot(a_axis, b_axis, label="Time-Acceleration")

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m^2/s)")

plt.grid(True)
plt.legend(loc="upper right")


plt.subplot(212)
plt.title("Time-Strain")
plt.plot(a_axis, c_axis, label="Time-Strain")

plt.xlabel("Time (s)")
plt.ylabel(r"$Strain (\mu m/m)$")

plt.grid(True)
plt.legend(loc="upper right")

plt.show()

#%%
# plt.figure(figsize=(16,12))
# plt.subplot(211)
# The above 2 lines equals to:
figure, axes = plt.subplots(2,1,)

axes[0].plot(a_axis, b_axis, label="Time-Acceleration")
axes[1].plot(a_axis, c_axis, label="Time-Strain")
axes[0].set_ylabel("Acceleration (m^2/s)")
axes[1].set_ylabel(r"$Strain (\mu m/m)$")

for axis in axes:
    axis.legend(loc="upper right")
    axis.grid()
    axis.set_xlabel("Time (s)")
    axis.set_title("common title")

#%%

plt.figure(figsize=(16,6))

plt.title("Time-Acceleration & Time-Velocity")
plt.plot(a_axis, b_axis, 'r-', linewidth=2,  label="Time-Acceleration")
plt.plot(a_axis, d_axis, color = 'cyan', linestyle = '--', linewidth=10, alpha=0.5, label="Time-Velocity")

plt.xlabel("Time (s)")
# plt.ylabel("Acceleration (m^2/s)")

plt.grid(True)
plt.legend()

plt.show()