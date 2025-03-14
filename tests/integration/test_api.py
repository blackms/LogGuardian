"""
Integration tests for LogGuardian API.
"""
import os
import time
import unittest
import json
import requests

# API URL from environment or default to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 10  # seconds


class TestLogGuardianAPI(unittest.TestCase):
    """
    Integration tests for LogGuardian API.
    """

    def setUp(self):
        """
        Setup before each test.
        """
        # Check if API is available
        retries = 3
        for i in range(retries):
            try:
                response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        self.fail(f"API is not available at {API_URL}")

    def test_health_endpoint(self):
        """
        Test health endpoint.
        """
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)
        self.assertIn("uptime", data)

    def test_detect_anomalies_simple(self):
        """
        Test anomaly detection with simple log sequences.
        """
        # Create sample logs
        logs = [
            {
                "message": "2023-02-15 10:12:34 INFO Server startup complete",
                "timestamp": "2023-02-15T10:12:34Z",
                "source": "test"
            },
            {
                "message": "2023-02-15 10:12:35 INFO Configuration loaded successfully",
                "timestamp": "2023-02-15T10:12:35Z",
                "source": "test"
            },
            {
                "message": "2023-02-15 10:12:40 ERROR Database connection failed",
                "timestamp": "2023-02-15T10:12:40Z",
                "source": "test"
            }
        ]
        
        # Send request
        request_data = {
            "logs": logs,
            "window_size": 3,
            "stride": 1,
            "batch_size": 1,
            "raw_output": True
        }
        
        response = requests.post(
            f"{API_URL}/detect",
            json=request_data,
            timeout=TIMEOUT
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertIn("inference_time", data)
        self.assertIn("processing_time", data)
        
        # Check results
        results = data["results"]
        self.assertEqual(len(results), 3)
        
        # Check that source and timestamp are preserved
        for i, result in enumerate(results):
            self.assertEqual(result["source"], logs[i]["source"])
            self.assertEqual(result["timestamp"], logs[i]["timestamp"])


if __name__ == "__main__":
    unittest.main()