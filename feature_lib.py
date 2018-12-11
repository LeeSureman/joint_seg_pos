def seenWord(state):
    assert len(state.word[-1]) == 1
    return 't1:' + state.word[-2]


def lastWordByWord(state):
    assert len(state.word[-1]) == 1
    return 't2:' + state.word[-3] + '_' + state.word[-2]


def oneCharWord(state):
    assert len(state.word[-1]) == 1
    assert len(state.word[-2]) == 1
    return 't3:' + state.word[-2]


def lengthByFirstChar(state):
    assert len(state.word[-1]) == 1
    return 't4:' + state.word[-2][0] + '_' + str(len(state.word[-2]))


def lengthByLastChar(state):
    assert len(state.word[-1]) == 1
    return 't5:' + state.word[-2][-1] + '_' + str(len(state.word[-2]))


def separateChars(state):
    assert len(state.word[-1]) == 1
    return 't6:' + state.word[-2][-1] + '_' + state.word[-1][0]


def consecutiveChars(state):
    assert len(state.word[-1]) > 1
    return 't7:' + state.word[-1][-2:]


def firstAndLastChars(state):
    assert len(state.word[-1]) == 1
    return 't8:' + state.word[-2][0] + '_' + state.word[-2][-1]


def lastWordFirstChar(state):
    assert len(state.word[-1]) == 1
    return 't9:' + state.word[-2] + '_' + state.word[-1][0]


def currentWordLastChar(state):
    assert len(state.word[-1]) == 1
    return 't10:' + state.word[-3][-1] + '_' + state.word[-2]


def firstCharLastWordByWord(state):
    assert len(state.word[-1]) == 1
    return 't11:' + state.word[-2][0] + '_' + state.word[-1][0]


def lastWordByLastChar(state):
    assert len(state.word[-1]) == 1
    return 't12:' + state.word[-3][-1] + '_' + state.word[-2][-1]


def lengthByLastWord(state):
    assert len(state.word[-1]) == 1
    return 't13:' + state.word[-3] + '_' + str(len(state.word[-2]))


def lastLengthByWord(state):
    assert len(state.word[-1]) == 1
    return 't14:' + str(len(state.word[-3])) + '_' + state.word[-2]


def currentTag(state):
    assert len(state.word[-1]) == 1
    return 't15:' + state.word[-2] + '_' + state.tag[-2]


def lastTagByTag(state):
    assert len(state.word[-1]) == 1
    try:
        r = 't16:' + state.tag[-2] + '_' + state.tag[-1]
    except TypeError as e:
        print(state.word[-2])
        print(state.word[-1])
        print(state.tag[-2])
        print(state.tag[-1])
    # return 't16:' + state.tag[-2] + '_' + state.tag[-1]
    return r

def lastTwoTagsByTag(state):
    assert len(state.word[-1]) == 1
    return 't17:' + state.tag[-3] + '_' + state.tag[-2] + '_' + state.tag[-1]


def tagByLastWord(state):
    assert len(state.word[-1]) == 1
    return 't18:' + state.word[-2] + '_' + state.tag[-1]


def lastTagByWord(state):
    assert len(state.word[-1]) == 1
    return 't19:' + state.tag[-3] + '_' + state.word[-2]


def tagByWordAndPrevChar(state):
    assert len(state.word[-1]) == 1
    return 't20:' + state.word[-3][-1] + '_' + state.word[-2] + '_' + state.tag[-2]


def tagByWordAndNextChar(state):
    assert len(state.word[-1]) == 1
    return 't21:' + state.word[-2] + '_' + state.tag[-2] + '_' + state.word[-1][0]


def tagOfOneCharWord(state):
    assert len(state.word[-1]) == 1
    assert len(state.word[-2]) == 1
    return 't22:' + state.word[-3][-1] + '_' + state.word[-2] + '_' + state.tag[-2] + '_' + state.word[-1][0]


def tagByFirstChar(state):
    assert len(state.word[-1]) == 1
    return 't23:' + state.word[-1][0] + '_' + state.tag[-1]


def tagByLastChar(state):
    assert len(state.word[-1]) == 1
    return 't24:' + state.word[-2][-1] + '_' + state.tag[-2]


def tagByChar(state):
    #be called either append or separate
    # seems like tagByFirstChar, but when they are called are diffenrent
    return 't25:' + state.word[-1][-1] + '_' + state.tag[-1]

def taggedCharByFirstChar(state):
    assert len(state.word[-1]) > 1
    return 't26:'+state.word[-1][0]+'_'+state.tag[-1]+'_'+state.word[-1][-1]


def taggedCharByLastChar(state):
    result = []
    for i in range(len(state.word[-2])-1):
        result.append('t27:'+state.tag[-2]+'_'+state.word[-2][i]+'_'+state.word[-2][-1])

    return result


def tagByFirstCharCat(state,tagset_hash):
    assert len(state.word[-1]) == 1
    return state.tag[-1]+'_'+tagset_hash


def tagByLastCharCat(state,tagset_hash):
    assert len(state.word[-1]) == 1
    return state.tag[-2]+'_'+tagset_hash

def taggedSeparateChars(state):
    assert len(state.word[-1]) == 1
    return 't30:'+state.word[-2][0]+'_'+state.tag[-2]+'_'+state.word[-1][0]+'_'+state.tag[-1]

def taggedConsecutiveChars(state):
    return 't31:'+state.word[-1][-2]+'_'+state.tag[-1]+'_'+state.word[-1][-1]