import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fmts_refpool/core'))
from compression_methods import (ContextCompressor,mean_llmlingua_word_labels,
 LLMLINGUA_WORD_SEPARATOR as SEP, LLMLINGUA_LABEL_SEPARATOR)
from turn_types import Turn

def labeled(words,labels):return SEP.join(f'{w} {l}' for w,l in zip(words,labels))
class TestLabels(unittest.TestCase):
 def test_content_digits_never_enter_average(self):
  self.assertEqual(mean_llmlingua_word_labels(labeled(['2026','12:45','$49.99','hotel'],[0,1,0,1])),.5)
 def test_unicode_and_spaces_in_word(self):
  self.assertEqual(mean_llmlingua_word_labels(labeled(['서울','word with space','ホテル'],[1,0,1])),2/3)
 def test_valid_zero_labels_are_not_error(self):
  self.assertEqual(mean_llmlingua_word_labels(labeled(['2026','east'],[0,0])),0)
 def test_actual_xlm_roberta_whitespace_word(self):
  # A real upstream result contains an empty surface word before punctuation.
  import json
  example=json.loads((Path(__file__).resolve().parent/'fixtures/llmlingua_whitespace_word.json').read_text())
  self.assertEqual(mean_llmlingua_word_labels(example['labeled_text']),16/37)
 def test_malformed_outputs_fail_visibly(self):
  for bad in [None,'','hotel','hotel 2','hotel 0.7','hotel 1'+SEP,'hotel true']:
   with self.subTest(bad=bad),self.assertRaises(ValueError):mean_llmlingua_word_labels(bad)
 def test_real_selector_numeric_substitution_invariant(self):
  class Stub:
   def compress_prompt(self,context,**kw):
    assert kw['word_sep']==SEP and kw['label_sep']==LLMLINGUA_LABEL_SEPARATOR
    return {'fn_labeled_original_prompt':labeled(context[0].split(),[0]*3)}
  turns=[Turn(speaker='USER',text='a hotel east'),Turn(speaker='USER',text='a hotel 2026')]
  with patch.object(ContextCompressor,'_llmlingua2_compressor',Stub()):
   self.assertEqual(ContextCompressor(.5).llmlingua2(turns).kept_turn_indices,[0])
 def test_retention_labels_choose_higher_scored_turn(self):
  class Stub:
   def compress_prompt(self,context,**kw):
    return {'fn_labeled_original_prompt':labeled(context[0].split(),[0,0] if '2026' in context[0] else [1,1])}
  turns=[Turn(speaker='USER',text='price 2026'),Turn(speaker='USER',text='hotel east')]
  with patch.object(ContextCompressor,'_llmlingua2_compressor',Stub()):
   self.assertEqual(ContextCompressor(.5).llmlingua2(turns).kept_turn_indices,[1])
 def test_inference_and_schema_error_abort_instead_of_zero(self):
  class Stub:
   def compress_prompt(self,*args,**kw):return {}
  with patch.object(ContextCompressor,'_llmlingua2_compressor',Stub()):
   with self.assertRaisesRegex(RuntimeError,'turn 0'):
    ContextCompressor(.5).llmlingua2([Turn(speaker='USER',text='hotel'),Turn(speaker='USER',text='east')])
 def test_empty_turn_policy_and_full_retention(self):
  with patch.object(ContextCompressor,'_get_llmlingua2',side_effect=AssertionError('unneeded inference')):
   self.assertEqual(ContextCompressor(1).llmlingua2([Turn(speaker='USER',text='a')]).kept_turn_indices,[0])
   self.assertEqual(ContextCompressor(.3).llmlingua2([]).kept_turn_indices,[])
if __name__=='__main__':unittest.main()
