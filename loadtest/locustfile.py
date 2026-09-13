import random

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.1, 1)

    @task
    def predict(self):
        pu_do = f"{random.randint(1, 265)}_{random.randint(1, 265)}"
        trip_distance = round(random.uniform(0.1, 50), 2)
        self.client.post("/predict", json={"pu_do": pu_do, "trip_distance": trip_distance})