import unittest
from detect_phishing import detect_phishing

class TestDetectPhishing(unittest.TestCase):
    def test_detect_phishing(self):
        text = "Your account has been compromised. Click here to reset your password."
        model_path = '../models/phishing_detector.pkl'
        result = detect_phishing(text, model_path)
        self.assertIn(result, [0, 1])

if __name__ == '__main__':
    unittest.main()
