import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("== /health ==")
print(client.get("/health").json())

weather24 = [{"hour": h, "solar_rad": 1.5 if 7 <= h <= 18 else 0.0, "temp": 20.0, "cloud": 3.0,
              "wind_speed": 6.0} for h in range(1, 25)]

print("\n== /predict (solar, valid) ==")
r = client.post("/predict", json={
    "energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15", "weather": weather24,
})
print(r.status_code)
body = r.json()
print("note:", body.get("note"))
print("hour 12:", [h for h in body["hourly"] if h["hour"] == 12])

print("\n== /predict (wind, valid, with demand) ==")
r = client.post("/predict", json={
    "energy_type": "wind", "region": "한림읍", "target_date": "2023-06-15", "weather": weather24,
    "demand_forecast_mw": [600.0] * 24,
})
print(r.status_code)
body = r.json()
print("hour 12:", [h for h in body["hourly"] if h["hour"] == 12])

print("\n== /predict (INVALID_HOUR_SET: 23개만 보냄) ==")
r = client.post("/predict", json={
    "energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15", "weather": weather24[:23],
})
print(r.status_code, r.json())

print("\n== /predict (INVALID_HOUR_SET: 중복 시간) ==")
bad = weather24[:23] + [weather24[0]]
r = client.post("/predict", json={
    "energy_type": "solar", "region": "남원읍", "target_date": "2023-06-15", "weather": bad,
})
print(r.status_code, r.json())

print("\n== /ess/simulate (hourly_capped) ==")
r = client.post("/ess/simulate", json={
    "hourly_curtailment_mwh": [10, 30, 5, 0, 40], "rated_power_mw": 22.5, "method": "hourly_capped",
})
print(r.status_code, r.json())
