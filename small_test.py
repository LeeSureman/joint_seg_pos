from ctb_preprocess import *
from feature_lib import *
from enum import Enum
import copy
import time
import pickle
from tagger import *
def compare(predict,gold):
    predict.word = predict.word[2:-1]
    predict.tag = predict.tag[2:-1]
    gold.word = gold.word[2:-1]
    gold.tag = gold.tag[2:-1]
    predict_all_num = len(predict.word)
    gold_all_num = len(gold.word)
    predict_correct_num = 0
    correct_joint = set()
    for i in range(len(gold.word)):
        correct_joint.add(gold.word[i]+gold.tag[i])

    for i in range(len(predict.word)):
        if predict.word[i]+predict.tag[i] in correct_joint:
            predict_correct_num+=1
    return predict_correct_num,predict_all_num,gold_all_num

f = open('trained_tagger_7.pkl','rb')
t = pickle.load(f)

train,dev,test = loadCTB3Data()

predict_all_num = 0
gold_all_num = 0
predict_correct_num = 0
for i in range(len(test[2])):
    rule = t.judge_by_rule(test[2][i])
    gold = State(test[0][i],test[1][i],True)
    predict = t.tag(test[2][i],False,rule,gold)
    p1,p2,p3 = compare(predict,gold)
    predict_all_num+=p2
    gold_all_num+=p3
    predict_correct_num+=p1

precision = predict_correct_num/predict_all_num
recall = predict_correct_num/gold_all_num
f_score = 2/(1/precision+1/recall)
print(f_score)
