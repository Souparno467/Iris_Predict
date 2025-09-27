import unittest
from app import app

class AppTestCase(unittest.TestCase):# Defined the test case class
    def setUp(self):# Set up the test client
        self.client = app.test_client()

    def test_home(self):# Test for the home endpoint
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Iris Predictor", resp.data)

    def test_api_predict(self):# Test for the /api/predict endpoint
        payload = {
            "sepal_length": 5.0,
            "sepal_width": 3.6,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        resp = self.client.post("/api/predict", json=payload)#  POST request sent with JSON payload
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("class_name", data)# Checking for expected keys in the response
        self.assertIn("probabilities", data)# Checking for expected keys in the response

if __name__ == "__main__":
    try:
        unittest.main(exit=False)  # tests passed
        print("Test cases passed successfully")
    except Exception as e: # tests failed
        print("Test cases failed")
        print("Reason:", e)
