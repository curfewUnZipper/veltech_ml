import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
N_SAMPLES = 5000

HEALTHY_RATIO = 0.7
DEGRADING_RATIO = 0.2
FAILING_RATIO = 0.1

# -----------------------------
# HELPERS
# -----------------------------
def add_noise(value, scale=0.02):
    return value + np.random.normal(0, scale)

def compute_voltage_sag(current, health_factor):
    """
    Simulate internal resistance effect.
    Worse battery → larger sag.
    """
    internal_resistance = np.random.uniform(0.015, 0.04) * (1 / health_factor)
    sag = current * internal_resistance
    return sag

def compute_dv_dt(health_factor):
    """
    Healthier battery → slower discharge
    """
    base = np.random.uniform(-0.015, -0.003)
    return base * (1 / health_factor)

# -----------------------------
# DATA GENERATION
# -----------------------------
data = []

n_healthy = int(N_SAMPLES * HEALTHY_RATIO)
n_degrading = int(N_SAMPLES * DEGRADING_RATIO)
n_failing = int(N_SAMPLES * FAILING_RATIO)

# =============================
# 🟢 HEALTHY BATTERY
# =============================
for _ in range(n_healthy):
    health_factor = np.random.uniform(0.85, 1.0)

    voltage_rest = np.random.uniform(12.4, 12.8)
    current = np.random.uniform(1, 8)
    temperature = np.random.uniform(20, 40)

    sag = compute_voltage_sag(current, health_factor)
    voltage_load = voltage_rest - sag

    dv_dt = compute_dv_dt(health_factor)

    data.append({
        "voltage_rest": add_noise(voltage_rest),
        "voltage_load": add_noise(voltage_load),
        "voltage_sag": add_noise(sag),
        "current": add_noise(current, 0.1),
        "temperature": add_noise(temperature, 0.5),
        "dv_dt": add_noise(dv_dt, 0.002),
        "label": 0
    })

# =============================
# 🟡 DEGRADING BATTERY
# =============================
for _ in range(n_degrading):
    health_factor = np.random.uniform(0.5, 0.85)

    voltage_rest = np.random.uniform(11.9, 12.4)
    current = np.random.uniform(3, 12)
    temperature = np.random.uniform(25, 50)

    sag = compute_voltage_sag(current, health_factor)
    voltage_load = voltage_rest - sag

    dv_dt = compute_dv_dt(health_factor)

    data.append({
        "voltage_rest": add_noise(voltage_rest),
        "voltage_load": add_noise(voltage_load),
        "voltage_sag": add_noise(sag),
        "current": add_noise(current, 0.2),
        "temperature": add_noise(temperature, 0.7),
        "dv_dt": add_noise(dv_dt, 0.003),
        "label": 0   # still not failed
    })

# =============================
# 🔴 FAILING BATTERY
# =============================
for _ in range(n_failing):
    health_factor = np.random.uniform(0.2, 0.5)

    voltage_rest = np.random.uniform(9.8, 11.9)
    current = np.random.uniform(5, 20)
    temperature = np.random.uniform(30, 60)

    sag = compute_voltage_sag(current, health_factor) * 1.5
    voltage_load = voltage_rest - sag

    dv_dt = compute_dv_dt(health_factor) * 2

    data.append({
        "voltage_rest": add_noise(voltage_rest),
        "voltage_load": add_noise(voltage_load),
        "voltage_sag": add_noise(sag),
        "current": add_noise(current, 0.3),
        "temperature": add_noise(temperature, 1.0),
        "dv_dt": add_noise(dv_dt, 0.005),
        "label": 1
    })

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(data)

# Shuffle like real telemetry
df = df.sample(frac=1).reset_index(drop=True)

# Save
df.to_csv("ev_12v_battery_dataset.csv", index=False)

print("✅ Dataset generated!")
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts(normalize=True))