import os
import xml.etree.ElementTree as ET

root = r"D:\yue zhang\ctb8.0\data\postagged"
# root = r'D:\重要的文件夹\ctb8.0\data\postagged'
file_path = os.listdir(root)
# print(file_path)
raw = []
segmented = []
tag = []
# f = open(root + '\\' + 'chtb_0930.nw.pos', encoding='utf-8')
sid = 0


def parseTagged(line, fp, i):
    line = line.strip()
    line = line.split(' ')
    s = []
    t = []
    try:
        for pair in line:
            p1, p2 = pair.split('_')
            s.append(p1)
            t.append(p2)
            # if '-' in p2:
            #     print(line)
    except ValueError as e:
        print(fp)
        print(i)
    return s, t


def loadGeneralData():
    for fp in file_path:
        if fp < r"chtb_1152":
            f = open(root + '\\' + fp, encoding='utf-8')
            lines = f.readlines()

            for i in range(len(lines)):
                if lines[i][0:4] == '</S>':
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    segmented.append(s)
                    tag.append(t)
                    raw.append(''.join(s))
        elif fp < r'chtb_3146':
            f = open(root + '\\' + fp, encoding='utf-8')
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][0] != '<':
                    s, t = parseTagged(lines[i], fp, i)
                    segmented.append(s)
                    tag.append(t)
                    raw.append(''.join(s))
        else:
            f = open(root + '\\' + fp, encoding='utf-8')
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][0] != '<':
                    if lines[i][0] == '\n':
                        continue
                    s, t = parseTagged(lines[i], fp, i)
                    segmented.append(s)
                    tag.append(t)
                    raw.append(''.join(s))

    return segmented, tag, raw


def loadCTB3Data():
    train_raw = []
    train_segmented = []
    train_tag = []
    dev_raw = []
    dev_segmented = []
    dev_tag = []
    test_raw = []
    test_segmented = []
    test_tag = []
    # load data that distributes like used in paper
    for fp in file_path:
        if (fp<r'chtb_0326'and fp>r'chtb_0301'):
            f = open(root + '\\' + fp, encoding='utf-8')
            lines = f.readlines()
            # print(len(lines))
            for i in range(len(lines)):
                # print(lines[i][0:4])
                if lines[i][0:4] == '</S>':
                    # print(1)
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    dev_segmented.append(s)
                    dev_tag.append(t)
                    dev_raw.append(''.join(s))
                    # print('dev:',len(dev_segmented))
                    # print('dev:',len(dev_tag))
                    # print('dev:',len(dev_raw))
        if (fp<r'chtb_0301'and fp>r'chtb_0271'):
            f = open(root + '\\' + fp, encoding='utf-8')
            lines = f.readlines()
            for i in range(len(lines)):
                if lines[i][0:4] == '</S>':
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    test_segmented.append(s)
                    test_tag.append(t)
                    test_raw.append(''.join(s))

        if fp < r"chtb_1152":
            f = open(root + '\\' + fp, encoding='utf-8')
            lines = f.readlines()

            for i in range(len(lines)):
                if lines[i][0:4] == '</S>':
                    line = lines[i - 1]
                    if line[0] == '<':
                        continue
                    s, t = parseTagged(line, fp, i)
                    train_segmented.append(s)
                    train_tag.append(t)
                    train_raw.append(''.join(s))
                    # print('train:',len(train_segmented))
                    # print('train:',len(train_tag))
                    # print('train:',len(train_raw))
    train=[train_segmented,train_tag,train_raw]
    dev=[dev_segmented,dev_tag,dev_raw]
    test=[test_segmented,test_tag,test_raw]
    # print(len(dev_segmented))
    # print(len(test_segmented))
    # print(dev_raw)
    print('load CTB3 successfully')
    return train,dev,test

if __name__ == '__main__':
    # result = loadCTB3Data()
    # print(len(result[0][0]))
    # char_set = set()
    # for s in result[0][2]:
    #     for c in s:
    #         char_set.add(c)
    #
    # print(len(char_set))
    result = loadGeneralData()
    char_set = set()
    for s in result[0]:
        for w in s:
            if w =='鸡':
                print(s)
    print(len(char_set))


# print(len(result[1][0]))
# print(len(result[2][0]))


# s, t, r = loadGeneralData()
# tagDict = {}
# wordDict = {}
# for tags in t:
#     for tag in tags:
#         if tagDict.get(tag) is None:
#             tagDict[tag] = 1
#         else:
#             tagDict[tag] += 1
#
# print(tagDict)
# print(len(tagDict))
#
# for sent in s:
#     for word in sent:
#         if wordDict.get(word) is None:
#             wordDict[word] = 1
#         else:
#             wordDict[word] += 1
#
# freq = list(wordDict.items())
# freq.sort(key=lambda x: x[1], reverse=True)
# # print(list(wordDict.items()).sort(key=lambda x:x[1],reverse=True))
# print(freq)
# print(len(wordDict))
