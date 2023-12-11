from scipy import stats

with open("Translation_Model/data_special/student_model/test_en.txt","r") as file:
    gt = file.readlines()

with open("Translation_Model/data_special/student_model/student_output.txt","r") as file:
    pred = file.readlines()

# with open("Translation_Model/data_special/student_model/test_de.txt","r") as file:
#     src = file.readlines()



substring = "BLEU"
indexes = [index for index, element in enumerate(pred) if substring not in element]

pred_filter = [element for index, element in enumerate(pred) if index not in indexes]
gt_filter = [element for index, element in enumerate(gt) if index not in indexes]


gt_bleu = []
pred_bleu = []
gt_seq = []
pred_seq = []
for line in gt_filter:
    idx = line.find("BLEU")
    gt_bleu.append(int(line[idx+4:-1]))
    gt_seq.append(line[:idx])

# print(gt_bleu)
for line in pred_filter:
    idx = line.find("BLEU")
    # print(line[idx+4:-1])
    pred_bleu.append(int(line[idx+4:-1]))
    pred_seq.append(line[:idx])


res = stats.pearsonr(gt_bleu,pred_bleu)

print(f"Pearson coorelation score for BLEU token is {round(res.statistic,3)}.")