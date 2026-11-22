import unittest
from tools.summarize66 import average, counts, decide, paired_ci, support
from tools.metrics59 import baseline_all_positive


def pair(id,text,spans,doc):
    return {'pair_id':id,'model_output':text,'spans':spans,'doc_ids':[doc]}


class Summary66Tests(unittest.TestCase):
    def test_tie_stability_and_average(self):
        a={'x':[{'start':0,'end':1,'risk':.5},{'start':3,'end':4,'risk':.7}]}
        b={'x':[{'start':0,'end':1,'risk':.9},{'start':3,'end':4,'risk':.1}]}
        c=average([{'v':a},{'v':b}],'v')
        self.assertAlmostEqual(c['x'][0]['risk'],.7)
        self.assertEqual(decide(c['x'],.5),[{'start':0,'end':1}])
        with self.assertRaises(ValueError): average([{'v':a},{'v':{'x':[]}}],'v')

    def test_support_agrees_with_all_positive_and_paired_ci(self):
        a=pair('a','one two three',[{'start':0,'end':2}],'d1')
        b=pair('b','four five',[],'d2')
        data=[a,b]
        choose={'a':[{'start':0,'end':12}],'b':[{'start':0,'end':8}]}
        self.assertAlmostEqual(support(data,choose)['f1'],baseline_all_positive(data)['f1'])
        ci=paired_ci(data,choose,rounds=100,seed=65)
        self.assertEqual((ci['low'],ci['high']),(0.,0.))
        self.assertEqual(ci['cluster_count'],2)

    def test_groups_are_documents(self):
        a=pair('a','one',[{'start':0,'end':2}],'shared')
        b=pair('b','two',[],'shared')
        c=pair('c','three',[],'other')
        ci=paired_ci([a,b,c],{'a':[{'start':0,'end':2}]},rounds=100,seed=65)
        self.assertEqual(ci['cluster_count'],2)
        self.assertEqual(ci,paired_ci([a,b,c],{'a':[{'start':0,'end':2}]},rounds=100,seed=65))

if __name__=='__main__':unittest.main()
