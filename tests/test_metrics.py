"""Hand-calculated metric tests; do not need model weights or a GPU."""
import ast
import math
from pathlib import Path
import unittest
import numpy as np
source=ast.parse((Path(__file__).resolve().parents[1]/'case_study.py').read_text())
functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in {'safe_ratio','binary_metrics','depth_metrics','pooled_metrics'}]
scope={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),'<metrics>','exec'),scope)
class MetricTests(unittest.TestCase):
    def test_confusion_counts(self):
        m=scope['binary_metrics'](np.array([1,1,0,0]),np.array([1,0,1,0]),np.ones(4,bool))
        self.assertEqual([m[k] for k in ['tp','fp','fn','tn']],[1,1,1,1])
        for k in ['precision','recall','fpr','f1']: self.assertEqual(m[k],.5)
    def test_undefined_ratios(self):
        m=scope['binary_metrics'](np.zeros(4),np.zeros(4),np.ones(4,bool))
        self.assertIsNone(m['precision']); self.assertIsNone(m['recall']); self.assertEqual(m['fpr'],0)
    def test_depth_errors(self):
        m=scope['depth_metrics'](np.array([2.,6.]),np.array([2.,4.]),np.ones(2,bool))
        self.assertEqual(m['mae'],1); self.assertAlmostEqual(m['rmse'],math.sqrt(2)); self.assertEqual(m['absrel'],.25)
    def test_masking(self):
        m=scope['depth_metrics'](np.array([99.,6.]),np.array([2.,4.]),np.array([False,True]))
        self.assertEqual(m['mae'],2);self.assertEqual(m['absrel'],.5)
    def test_nonfinite_output_rejected(self):
        with self.assertRaises(ValueError): scope['depth_metrics'](np.array([np.nan]),np.array([1.]),np.ones(1,bool))
    def test_pooled_counts(self):
        m=scope['pooled_metrics']([dict(tp=1,fp=0,fn=0,tn=9),dict(tp=1,fp=2,fn=2,tn=5)])
        self.assertEqual(m['tp'],2);self.assertEqual(m['precision'],.5);self.assertEqual(m['recall'],.5)
if __name__=='__main__':unittest.main()
