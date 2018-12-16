from ctb_preprocess import *
from feature_lib import *
from enum import Enum
import copy
import time
import pickle
from tagger import *
def countCorrect(predict,gold):
    predict.word = predict.word[2:-1]
    predict.tag = predict.tag[2:-1]
    gold.word = gold.word[2:-1]
    gold.tag = gold.tag[2:-1]
    predict_all_num = len(predict.word)
    gold_all_num = len(gold.word)
    seg_correct_num = 0
    joint_correct_num = 0
    gold_index = 0
    gold_charindex = 0
    predict_index = 0
    predict_charindex = 0
    while gold_index<gold_all_num and predict_index<predict_all_num:
        if predict.word[predict_index] == gold.word[gold_index]:
            seg_correct_num+=1
            if predict.tag[predict_index] == gold.tag[gold_index]:
                joint_correct_num+=1
            predict_charindex+=len(predict.word[predict_index])
            gold_charindex+=len(gold.word[gold_index])
            predict_index+=1
            gold_index+=1
        else:
            if predict_charindex == gold_charindex:
                predict_charindex += len(predict.word[predict_index])
                gold_charindex += len(gold.word[gold_index])
                predict_index += 1
                gold_index += 1
            elif predict_charindex<gold_charindex:
                predict_charindex += len(predict.word[predict_index])
                predict_index+=1
            elif predict_charindex>gold_charindex:
                gold_charindex+=len(gold.word[gold_index])
                gold_index+=1

    return seg_correct_num,joint_correct_num,predict_all_num,gold_all_num




# f = open('trained_tagger_7.pkl','rb')
# t = pickle.load(f)
#
# train,dev,test = loadCTB3Data()
#
# predict_all_num = 0
# gold_all_num = 0
# predict_correct_num = 0
# for i in range(len(test[2])):
#     rule = t.judge_by_rule(test[2][i])
#     gold = State(test[0][i],test[1][i],True)
#     predict = t.tag(test[2][i],False,rule,gold)
#     p1,p2,p3 = compare(predict,gold)
#     predict_all_num+=p2
#     gold_all_num+=p3
#     predict_correct_num+=p1
#
# precision = predict_correct_num/predict_all_num
# recall = predict_correct_num/gold_all_num
# f_score = 2/(1/precision+1/recall)
# print(f_score)
p_word_raw = '的小狗-我们汽车-楼的-你'
g_word_raw = '的小狗我-们汽-车楼的-你'
# p_word = ['的','小狗','我们','汽车','楼']
# g_word = ['的小','狗我','们','汽车','楼']
p_word = p_word_raw.split('-')
g_word = g_word_raw.split('-')
# p_tag = ['N','N','N','N','N']
# g_tag = ['N','N','N','N','N']
p_tag = ['N']*len(p_word)
g_tag = ['N']*len(g_word)
predict = State(p_word,p_tag,True)
gold = State(g_word,g_tag,True)
print(countCorrect(predict,gold))