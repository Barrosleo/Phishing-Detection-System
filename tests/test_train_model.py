import unittest
from train_model import train_model

class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        model = train_model('../data/preprocessed_features.csv', '../data/preprocessed_labels.csv')
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
