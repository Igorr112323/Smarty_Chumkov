import unittest
from tools.recover66_summary import failure_is_after_cleanup


class RecoveryTests(unittest.TestCase):
    def sample(self):
        return {'outer_fold':1,'inner_fold':2,'seed':0,'status':'падение','stage':'predictions',
                'weights_package_bytes':123,
                'error':'File ".../tools/train66.py", line 106, in main\n tmp.cleanup()\nAttributeError: \'str\' object has no attribute \'cleanup\''}

    def test_known_error_only(self):
        r=self.sample(); self.assertTrue(failure_is_after_cleanup(r))
        for k,v in [('outer_fold',2),('seed',1),('stage','segment_training'),('weights_package_bytes',0),('error','ошибка модели')]:
            copy=dict(r);copy[k]=v;self.assertFalse(failure_is_after_cleanup(copy))

if __name__=='__main__':unittest.main()
