Data Files:

> bleu_scores_r4: BLEU scores for (teacher_model_outputs_de, references_en) round to four digits.
> 
> BLEU_tokens: BLEU tokens with 10 categories
> 
> bleu_tokens_c5: BLEU tokens with 5 categories
> 
> Targets_T.pth: generated translation from teacher model for train_data
> 
> Some other files used are in the main directory (like )

Data Processing colab 1.2.ipynb (In main directory)

> To do some basic preprocessing and to generate small dataset

Teacher_Model_Multi30k.ipynb:

> Notebook for teacher model

Student_Model_Attention.ipynb

> Notebook for student model
> 
> Here using Multi30k for convenience.
> 
> Please see Evaluation part.
> 
> The accuracy that [generated QE tokens from student model] vs [QE score for student translations] in test data is 0.439, which is not very good but better than guessing. (5 categories)
