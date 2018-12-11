from ctb_preprocess import *
from feature_lib import *
from enum import Enum
import copy
import time
import pickle

class Action(Enum):
    APPEND = 0
    SEPARATE = 1
    No = 2


class State(object):

    def __init__(self, word=None, tag=None, isGold=False):
        if word is None:
            word = []
        if tag is None:
            tag = []
        self.tag = tag
        self.word = word
        self.score = 0
        assert len(tag) == len(word)
        self.charLen = sum(map(lambda x: len(x), word))
        self.word.insert(0, ' ')
        self.tag.insert(0, 'PAD')
        self.word.insert(1, ' ')
        self.tag.insert(1, 'PAD')
        if isGold:
            self.word.append(' ')
            self.tag.append('PAD')

    def __eq__(self, other):
        for i in range(len(self.word)):
            if self.word[i] != other.word[i] or self.tag[i] != other.tag[i]:
                return False
            # try:
            #     if self.word[i] != other.word[i] or self.tag[i] != other.tag[i]:
            #         return False
            # except IndexError as e:
            #     print(self.word)
            #     print(self.tag)
            #     print(other.word)
            #     print(other.tag)
        return True

    def follow(self, gold):
        l = self.nowLen()
        wl = self.nowWordLen()
        # print('l:', l)
        # if l == 0:
        #     self.word.append(gold.word[0][0])
        #     self.tag.append(gold.tag[0])
        #     self.charLen += 1
        #     return
        if len(gold.word[l - 1]) > wl:
            self.word[-1] = gold.word[l - 1][0:wl + 1]
            self.charLen += 1
            return Action.APPEND
        elif len(gold.word[l - 1]) == wl:
            self.word.append(gold.word[l][0])
            self.tag.append(gold.tag[l])
            self.charLen += 1
            return Action.SEPARATE

    def nowLen(self):  # num of words
        return len(self.word)

    def nowWordLen(self):  # len of now word
        return len(self.word[-1])

    def append(self, c):
        new_state = copy.deepcopy(self)
        new_state.charLen += 1
        new_state.word[-1] += c
        return new_state

    def separate(self, c, t):
        new_state = copy.deepcopy(self)
        new_state.word.append(c)
        new_state.tag.append(t)
        new_state.charLen += 1
        return new_state


class AgendaBeam(object):
    def __init__(self, n=16):
        self.old = []
        self.new = []

    def update(self):
        self.old = self.new
        self.new.clear()

    def squeeze(self, n):
        self.old.sort(key=lambda x: x.score, reverse=True)
        if n > len(self.old):
            n = len(self.old)
        signatures = {}
        self.old = self.old[0:n]
        for old_state in self.old:
            if signatures.get(old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]) is None:
                signatures[old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]] = old_state
            else:
                if signatures[
                    old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]].score < old_state.score:
                    signatures[old_state.word[-2] + '_' + old_state.tag[-2] + '_' + old_state.tag[-1]] = old_state.score
        self.old = list(signatures.values())
        self.old.sort(key=lambda x: x.score, reverse=True)
        # print('len after squeeze: ',len(self.old))

    def clear(self):
        self.old = []
        self.new = []

class Weight(object):
    def __init__(self,i_round,value=0):
        self.now = value
        self.accumulated = 0
        self.used = 0
        self.last_update = i_round
    def accumulate(self,i_round):
        self.accumulated+=(i_round-self.last_update)*self.now
        self.last_update = i_round

    def update(self,amount,i_round):
        self.accumulate(i_round)
        self.now+=amount

    def useAverage(self,n_iterations):
        self.used = self.accumulated/n_iterations

    def useRaw(self):
        self.used = self.now

class FeatureWeight(object):
    # [newest_weight,weight_sum,weight_tmp_avg]
    def __init__(self):
        self.weightDict = {}

    def getFeatureScore(self, feature, isTrain):
        r = self.weightDict.get(feature)
        if r is None:
            return 0
        if isTrain:
            return r.now
        else:
            return r.used

    def updateFeatureScore(self, feature, amount, i_round):
        weight = self.weightDict.get(feature)
        if weight:
            weight.update(amount,i_round)
        else:
            self.weightDict[feature] = Weight(i_round,amount)

    # def accumulateWeight(self):
    #     for value in self.weightDict.values():
    #         value[1]+=value[0]


    # def getAverageWeight(self,iterations):
    #     for value in self.weightDict.values():
    #         value[2] = value[1]/iterations



class Tagger(object):
    def __init__(self):
        self.agenda = AgendaBeam(16)
        self.weight = FeatureWeight()
        self.word2tags = {}
        self.char2tags = {}
        tmp = set()
        tmp.add('PAD')
        self.char2tags[' '] = tmp
        self.char2tags_hash = {}
        self.frequent_word = set()
        self.word_frequency = {}
        self.max_frequency = 0
        self.tag2length = {}
        self.tag_set = set()
        # self.tag_set.add('PAD')
        self.train_rules = []
        self.char_can_start = set()
        self.threshold = 100

    def prepareKnowledge(self, training_set):
        for i in range(len(training_set[0])):
            for j in range(len(training_set[0][i])):
                tmp = self.word2tags.get(training_set[0][i][j])
                if tmp is None:
                    tmp = self.word2tags[training_set[0][i][j]] = set()
                    tmp.add(training_set[1][i][j])
                else:
                    tmp.add(training_set[1][i][j])
                for c in training_set[0][i][j]:
                    tmp = self.char2tags.get(c)
                    if tmp is None:
                        tmp = self.char2tags[c] = set()
                        tmp.add(training_set[1][i][j])
                    else:
                        tmp.add(training_set[1][i][j])
                # self.word2tags[training_set[0][i][j]] = training_set[1][i][j]
                self.char_can_start.add(training_set[0][i][j][0])
                if self.word_frequency.get(training_set[0][i][j]) is None:
                    self.word_frequency[training_set[0][i][j]] = 1
                else:
                    self.word_frequency[training_set[0][i][j]] += 1

                if self.tag2length.get(training_set[1][i][j]) is None:
                    self.tag_set.add(training_set[1][i][j])
                    self.tag2length[training_set[1][i][j]] = len(training_set[0][i][j])
                elif self.tag2length[training_set[1][i][j]] < len(training_set[0][i][j]):
                    self.tag2length[training_set[1][i][j]] = len(training_set[0][i][j])

            self.train_rules.append(self.judge_by_rule(training_set[2][i]))

        threshold = int(max(self.word_frequency.values()) / 5000) + 5
        self.threshold = threshold
        for pair in self.word_frequency.items():
            if pair[1] > threshold:
                self.frequent_word.add(pair[0])
        self.tag_list = list(self.tag_set)
        self.tag_list.sort()
        self.hash_char2tags()

    def hash_tags(self,tags):
        result = ['0']*(len(self.tag_list)+1)# PAD not in tag_set
        for i in range(len(self.tag_list)):
            if self.tag_list[i] in tags:
                result[i] = '1'
        if 'PAD' in tags:
            result[-1] = '1'
        return ''.join(result)

    def hash_char2tags(self):
        for pair in self.char2tags.items():
            self.char2tags_hash[pair[0]] = self.hash_tags(pair[1])


    def judge_by_rule(self, sentence):
        actions = []
        actions.append(Action.SEPARATE)
        for i in range(1, len(sentence)):
            if sentence[i - 1:i + 1].encode('utf-8').isdigit():
                actions.append(Action.APPEND)
                continue
            if sentence[i - 1:i + 1].encode('utf-8').isalpha():
                actions.append(Action.APPEND)
                continue
            if sentence[i - 1].encode('utf-8').isdigit() != sentence[i].encode('utf-8').isdigit():
                actions.append(Action.SEPARATE)
                continue
            if sentence[i - 1].encode('utf-8').isalpha() != sentence[i].encode('utf-8').isalpha():
                actions.append(Action.SEPARATE)
                continue
            actions.append(Action.No)
        return actions

    def tag(self, sentence, isTrain, rule, i_round,gold=None,sentence_index = 0,f=None):
        s = sentence
        self.agenda.old = [State()]
        goldFollow = State()
        for i in range(len(sentence)):
            if len(self.agenda.old) == 0:
                print('hasnot state return 1')
                return 1
            anyCorrect = False
            # try:
            #     assert len(self.agenda.new) == 0
            # except AssertionError
            if isTrain:
                for state in self.agenda.old:
                    if state == goldFollow:
                        anyCorrect = True
                        break

                if not anyCorrect:
                    self.updateScoreForState(self.agenda.old[0], -1,i_round)
                    self.updateScoreForState(goldFollow, 1,i_round)
                    if f:
                        f.write(str(sentence_index)+'th s,early update return 1\n')
                        print(sentence_index,'th s,early update return 1')
                        f.write('predict:\n')
                        print('predict:\n',end='')
                        # try:
                        for w in self.agenda.old[0].word:
                            f.write(w+'-')
                            print(w,end='-')
                        print('')
                        f.write('\n')
                        for t in self.agenda.old[0].tag:
                            f.write(t+'-')
                            print(t,end='-')
                        print('')
                        f.write('\n')
                        f.write('goldFollow:\n')
                        print('goldFollow:\n',end='')
                        for w in goldFollow.word:
                            f.write(w+'-')
                            print(w,end='-')
                        print('')
                        f.write('\n')
                        for t in goldFollow.tag:
                            f.write(t+'-')
                            print(t,end='-')
                        print('')
                        f.write('\n')
                        f.write('gold:\n')
                        print('gold:\n',end='')
                        for w in gold.word:
                            f.write(w+'-')
                            print(w,end='-')
                        print('')
                        f.write('\n')
                        for t in gold.tag:
                            f.write(t+'-')
                            print(t,end='-')
                        f.write('\n')
                        print('')

                    # except IndexError as e:
                    #     print('old len:',len(self.agenda.old))
                    #     exit(1)
                    self.agenda.clear()
                    return 1

            if rule[i] == Action.No:
                for old_state in self.agenda.old:
                    if  self.tag2length[old_state.tag[-1]] > len(old_state.word[-1]):
                        appended = old_state.append(sentence[i])
                        appended.score += self.getOrUpdateAppendScore(appended,isTrain)
                        self.agenda.new.append(appended)
                    if sentence[i] in self.char_can_start and self.canAssignTag(old_state.word[-1], old_state.tag[-1]):
                        for tag in self.tag_set:
                            canStart = False
                            for j in range(1,self.tag2length[tag]+1):
                                if self.word2tags.get(sentence[i:i + j]) and tag in self.word2tags[sentence[i:i + j]]:
                                    canStart = True
                                    break
                            if canStart == False:
                                continue
                            separated = old_state.separate(sentence[i], tag)
                            separated.score += self.getOrUpdateSeparateScore(separated,isTrain)
                            self.agenda.new.append(separated)

            elif rule[i] == Action.SEPARATE:
                for old_state in self.agenda.old:
                    for tag in self.tag_set:
                        canStart = False
                        for j in range(1,self.tag2length[tag]+1):
                            if self.word2tags.get(sentence[i:i + j]) and tag in self.word2tags[sentence[i:i + j]]:
                                canStart = True
                                break
                        if canStart == False:
                            continue
                        separated = old_state.separate(sentence[i], tag)
                        separated.score += self.getOrUpdateSeparateScore(separated,isTrain)
                        self.agenda.new.append(separated)
            elif rule[i] == Action.APPEND:
                for old_state in self.agenda.old:
                    appended = old_state.append(sentence[i])
                    appended.score += self.getOrUpdateAppendScore(appended,isTrain)
                    self.agenda.new.append(appended)
            if len(self.agenda.new) == 0:
                print(rule[i])
                print(sentence)
                print(sentence_index)
            self.agenda.old = self.agenda.new
            self.agenda.new = []
            # print('old:')
            # for w in self.agenda.old:
            #     print(w.word)
            self.agenda.squeeze(16)
            goldFollow.follow(gold)

        max_index = 0
        for i in range(len(self.agenda.old)):
            self.agenda.old[i] = self.agenda.old[i].separate(' ', 'PAD')
            self.agenda.old[i].score += self.getOrUpdateSeparateScore(self.agenda.old[i],isTrain)
            if self.agenda.old[i].score > self.agenda.old[max_index].score:
                max_index = i

        if isTrain:
            if self.agenda.old[max_index] == gold:
                f.write('predict right return 0\n')
                print('predict right return 0',sentence)
                f.write('predict:\n')
                print('predict:\n', end='')
                # try:
                for w in self.agenda.old[max_index].word:
                    f.write(w + '-')
                    print(w, end='-')
                f.write('\n')
                print('')
                for t in self.agenda.old[max_index].tag:
                    f.write(t + '-')
                    print(t, end='-')
                print('')
                f.write('\n')
                f.write('gold:\n')
                print('gold:\n', end='')
                for w in gold.word:
                    f.write(w + '-')
                    print(w, end='-')
                print('')
                f.write('\n')
                for t in gold.tag:
                    f.write(t + '-')
                    print(t, end='-')
                f.write('\n')
                print('')
                return 0
            else:
                self.updateScoreForState(gold, 1,i_round)
                self.updateScoreForState(self.agenda.old[max_index], -1,i_round)
                f.write('predict wrong return 1\n')
                print('predict wrong return 1')
                f.write('predict:\n')
                print('predict:\n', end='')
                # try:
                for w in self.agenda.old[max_index].word:
                    f.write(w + '-')
                    print(w, end='-')
                f.write('\n')
                print('')
                for t in self.agenda.old[max_index].tag:
                    f.write(t + '-')
                    print(t, end='-')
                print('')
                f.write('\n')
                f.write('gold:\n')
                print('gold:\n', end='')
                for w in gold.word:
                    f.write(w + '-')
                    print(w, end='-')
                f.write('\n')
                print('')
                for t in gold.tag:
                    f.write(t + '-')
                    print(t, end='-')
                f.write('\n')
                print('')
                self.agenda.clear()
                return 1
        else:
            return self.agenda.old[max_index]

    def updateScoreForState(self, state, amount,i_round):
        tmp_state = State()
        for i in range(state.charLen):
            action = tmp_state.follow(state)
            if action == Action.APPEND:
                self.getOrUpdateAppendScore(tmp_state,True, amount,i_round)
            elif action == Action.SEPARATE:
                self.getOrUpdateSeparateScore(tmp_state,True, amount,i_round)

        if tmp_state != state:
            raise ('follow not finished!')

    def getOrUpdateSeparateScore(self, state,isTrain,amount=0,i_round=0):
        assert len(state.word[-1]) == 1
        features = []
        features.append(seenWord(state))
        features.append(lastWordByWord(state))
        if len(state.word[-2]) == 1:
            features.append(oneCharWord(state))
        features.append(lengthByFirstChar(state))
        features.append(lengthByLastChar(state))
        features.append(separateChars(state))
        features.append(firstAndLastChars(state))
        features.append(lastWordFirstChar(state))
        features.append(currentWordLastChar(state))
        features.append(firstCharLastWordByWord(state))
        features.append(lastWordByLastChar(state))
        features.append(lengthByLastWord(state))
        features.append(lastLengthByWord(state))
        features.append(currentTag(state))
        features.append(lastTagByTag(state))
        features.append(lastTwoTagsByTag(state))
        features.append(tagByLastWord(state))
        features.append(lastTagByWord(state))
        features.append(tagByWordAndPrevChar(state))
        features.append(tagByWordAndNextChar(state))
        features.append(tagByFirstChar(state))
        features.append(tagByLastChar(state))
        features.append(tagByChar(state))
        features.extend(taggedCharByLastChar(state))  # return a list
        features.append(taggedSeparateChars(state))
        features.append(tagByFirstCharCat(state,self.char2tags_hash[state.word[-1]]))
        features.append(tagByLastCharCat(state,self.char2tags_hash[state.word[-2][-1]]))
        if amount == 0:
            score = 0
            for feature in features:
                score += self.weight.getFeatureScore(feature,isTrain)
            return score
        else:
            for feature in features:
                self.weight.updateFeatureScore(feature, amount,i_round)

    def getOrUpdateAppendScore(self, state,isTrain, amount=0,i_round=0):
        features = []
        features.append(consecutiveChars(state))
        features.append(tagByChar(state))
        features.append(taggedCharByFirstChar(state))
        features.append(taggedConsecutiveChars(state))
        if amount == 0:
            score = 0
            for feature in features:
                score += self.weight.getFeatureScore(feature,isTrain)
            return score
        else:
            for feature in features:
                self.weight.updateFeatureScore(feature, amount,i_round)

    def canAssignTag(self, word, tag):
        if word not in self.frequent_word:
            return True
        else:
            if tag in self.word2tags[word]:
                return True
            else:
                return False

    def train(self, training_set, iterations):
        f_weight = open('trained_tagger.pkl','wb')
        f_error = open('error_record.txt','a',encoding='utf-8')
        f_training_process = open('training_record.txt','a',encoding='utf-8')
        n_error = 0
        golds = []
        for i in range(len(training_set[2])):
            golds.append(State(training_set[0][i], training_set[1][i], True))
        old_time = time.time()
        for j in range(iterations):
            n_error = 0
            f_weight = open('trained_weight_'+str(j)+'.pkl', 'wb')
            f_error = open('error_record.txt', 'a',encoding='utf-8')
            f_training_process = open('training_record.txt', 'a',encoding='utf-8')
            for i in range(len(training_set[2])):
                n_error += self.tag(training_set[2][i], True, self.train_rules[i],j,
                                    gold=golds[i],sentence_index=i,f=f_training_process)
            f_error.write(str(j)+'th error: '+str(n_error)+' / '+str(len(training_set[2]))+'\n')
            print(j,'th error: ', n_error, ' / ', len(training_set[2]))
            new_time = time.time()
            f_error.write('time_spent:' + str(new_time - old_time) + '\n')
            print('time_spent:',new_time-old_time)
            old_time = new_time
            pickle.dump(self.weight.weightDict,f_weight)
            f_weight.close()
            f_error.close()
            f_training_process.close()

if __name__ == '__main__':
    train,dev,test = loadCTB3Data()
    # train[0][0],train[1][0],train[2][0] = train[0][5665],train[1][5665],train[2][5665]
    # print(train[0][0])
    t = Tagger()
    # rule = t.judge_by_rule('图为B&Q家饰五金量贩店一景。')
    #
    # for i in rule:
    #     print(i.value,end=' ')
    # print('')
    t.prepareKnowledge(train)
    t.train(train,40)
    # print(t.tag_list)
    # while(True):
    #     s = input()
    #     try:
    #         print(t.char2tags[s])
    #         print(t.char2tags_hash[s])
    #     except KeyError as e:
    #         continue
    # print(t.tag_set)
    # t.train(train,40)
    # for i in range(10):
    #     if i == 3:
    #         a = Weight(3, 4)
    #     if i >=3 and i%2 == 0:
    #         a.update(-1,i)
    #         print(i,'th w:',a.now,' accumulate:',a.accumulated)
# for i in rule:
#     print(i.value)
# train_seg = [['我', 'abc', '12'], ['1232', '李孝男']]
# train_tag = [['N', 'N', 'V'], ['Name', 'NickName']]
# train_raw = ['我abc12', '1234李孝男']
# t = Tagger()
# t.prepareKnowledge([train_seg, train_tag, train_raw])
# print(t.train_rules)
#
# l = State(['我', '李孝男', '打钱'], ['N', 'N', 'V'])
# print(l.word)
# print(l.tag)
# x = l.separate('啊', 'N')
# print(x.word)
# print(x.tag)
#
# a = AgendaBeam()
# s1 = State(['abc', 'de'], ['G', 'H'])
# s2 = State(['abc', 'de'], ['G', 'H'])
# s3 = State(['ac', 'de'], ['G', 'H'])
# s1.score = 1
# s2.score = 3
# s3.score = 0
# a.old = [s1, s2, s3]
# a.squeeze(3)
# print(a.old[0].word)
# print(a.old[0].tag)
# print(a.old[0].score)
# print(a.old[1].word)
# print(a.old[1].tag)
# print(a.old[1].score)

# t = Tagger()


# word = ['西电','da帅哥','李磊']
# tag = ['NA','NB','NC']
# gold = State(word,tag)
# sub = State()
# print(gold.word)
# print(gold.tag)
# print(gold.charLen)
# while sub.charLen!=gold.charLen:
#     sub.follow(gold)
#     print(sub.word)
#     print(sub.tag)
#     print('charlen:',sub.charLen)
#     print('123456')
# print(sub==gold)


# t = Tagger()
# train,dev,test = loadCTB3Data()
# t.prepareKnowledge(train)
# print(len(t.char_can_start))
# word_freq = list(t.word_frequency.items())
# word_freq.sort(key=lambda x:x[1],reverse=True)
# print(word_freq)
