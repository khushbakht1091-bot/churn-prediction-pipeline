from locust import HttpUser, task, between

class ChurnUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_churn(self):
        payload = {
            "tenure": 12,
            "MonthlyCharges": 65.5,
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "TotalCharges": 786.0
        }

        self.client.post("/predict", json=payload)