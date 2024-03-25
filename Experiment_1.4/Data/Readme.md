
**bleu_tokens_c10_multi30k_de_en.pth**: 
dictionary {'train': tokens, 'test': tokens, 'valid': tokens}
Isometric Segmentation type category

**bleu_scores_r4_multi30k_de_en.pth**:
dictionary {'train': scores, 'test': scores, 'valid': scores}
BLEU scores from [Teacher Outputs, Reference] round to 4 digits

**Teacher_Translations_multi30k_de_en.pth**
dictionary {'train': translations, 'test': translations, 'valid': translations}
Teacher Outputs, Trg_T

**bleu_SR_train**
BLEU scores from [Student Outputs, Reference] round to 4 digits, only for train set

**bleu_SR_test**
BLEU scores from [Student Outputs, Reference] round to 4 digits, only for test set

**train_outputs**
Student Outputs (trained by references, without BLEU token), Trg_S