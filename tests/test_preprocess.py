import unittest
from preprocess import preprocess_data

class TestPreprocess(unittest.TestCase):
    def test_preprocess_data(self):
        features, labels = preprocess_data('../data/emails.csv')
        self.assertIsNotNone(features)
        self.assertIsNotNone(labels)

if __name__ == '__main__':
    unittest.main()
