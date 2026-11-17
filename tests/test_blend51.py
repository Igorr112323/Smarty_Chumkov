import unittest
from tools.blend51 import blend, calibrate, fit_isotonic


class Blend51Test(unittest.TestCase):
    def test_different_subword_offsets(self):
        a = {'answer':'abcdef', 'pieces': [[0,3,0.8],[3,6,0.2]]}
        b = {'answer':'abcdef', 'pieces': [[0,2,0.4],[2,6,0.6]]}
        out = blend(a,b)['pieces']
        self.assertAlmostEqual(out[0][2], (0.8 + (2*0.4+0.6)/3)/2)
        self.assertAlmostEqual(out[1][2], (0.2+0.6)/2)
        with self.assertRaises(ValueError):
            blend(a, {'answer':'wrong','pieces':[]})
        with self.assertRaises(ValueError):
            blend(a, {'answer':'abcdef','pieces':[[0,2,0.3],[0,3,0.3]]})

    def test_isotonic_training_and_interpolation(self):
        model=fit_isotonic([(0.1,1),(0.1,0),(0.3,0),(0.9,1)])
        self.assertEqual(model[0]['value'], 1/3)
        self.assertEqual(calibrate(0.1,model), 1/3)
        self.assertEqual(calibrate(0.3,model), 1/3)
        self.assertEqual(calibrate(0.9,model), 1.0)
        self.assertLess(calibrate(0.4,model),calibrate(0.7,model))


if __name__ == '__main__':
    unittest.main()
