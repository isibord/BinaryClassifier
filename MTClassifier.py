import sys
import nltk
import random
import math

class Translated:

    def __init__(self, refSent, canSent, score, translatedBy):
        self.refSent = refSent #reference sentence
        self.canSent = canSent #candidate sentence
        self.score = score
        self.translatedBy = translatedBy #translated by human H, machine M, or unknown ?

def writeScoresToFile(humanScores, machineScores):
    with open("humanscores.csv", 'w', encoding='utf8') as fw:
        for score in humanScores:
            fw.write(str(score))
            fw.write('\n')
    with open("machinescores.csv", 'w', encoding='utf8') as fw:
        for score in machineScores:
            fw.write(str(score))
            fw.write('\n')

def main(filepath, filepath_predict):
    print('about to read input file...')
    trainFile, humanScores, machineScores = readFileForTraining(filepath)
    #writeScoresToFile(humanScores, machineScores)
    classifier, word_features = trainClassifier(trainFile)
    ref, pred = predictFromFile(filepath_predict, classifier, word_features)
    #evaluateAccuracy(ref, pred)


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def trainClassifier(trainObj):
    l = []

    for item in trainObj:
        l = l + [w for w in item.canSent.split()]
        l = l + [str(round(item.score, 1))]
        #l = l + [w for w in item.refSent.split()]
    l = set(l)

    all_words = nltk.FreqDist(l)
    word_features = all_words.keys()
    train_set = []

    for item in trainObj:
        mylist = [w for w in item.canSent.split()]
        train_set.append((document_features(mylist, word_features), item.translatedBy))
        mylist = [str(round(item.score, 1))]
        train_set.append((document_features(mylist, word_features), item.translatedBy))

        #mylist = [w for w in item.refSent.split()]
        #train_set.append((document_features(mylist, word_features), 'H'))

    random.shuffle(train_set)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return classifier, word_features


def predictSent(refSent, canSent, score, classifier, word_features):
    temp = canSent.split()
    mylist = [w for w in temp]
    mylist = mylist + [str(round(score, 1))]
    prediction = classifier.classify(document_features(mylist, word_features))
    #print(prediction)
    return prediction

def readFileForTraining(filepath):
    trackRows = 1
    refSent = ''
    canSent = ''
    score = 0.0
    transBy = ''
    listTranslated = []
    humanScores = []
    machineScores = []
    with open(filepath, 'r', encoding='utf8') as f:
        for row in f:
            currRow = row.rstrip()
            if not currRow:
                # new group start
                trackRows = 1
                refSent = ''
                canSent = ''
                score = 0.0
                transBy = ''
            else:
                if trackRows == 2:
                    refSent = currRow
                elif trackRows == 3:
                    canSent = currRow
                elif trackRows == 4:
                    score = float(currRow)
                elif trackRows == 5:
                    transBy = currRow
                    newObj = Translated(refSent, canSent, score, transBy)
                    listTranslated.append(newObj)
                    if transBy == 'H':
                        humanScores.append(score)
                    else:
                        machineScores.append(score)
                elif trackRows == 1:
                    # chinese sentence, don't care for it
                    trackRows += 1
                    continue
                trackRows += 1
    return listTranslated, humanScores, machineScores

def predictFromFile(filepath_predict, classifier, word_features):
    trackRows = 1
    refSent = ''
    canSent = ''
    score = 0.0
    list_ref = []
    list_predict = []
    with open(filepath_predict.replace('unlabeled', 'predicted'), 'w', encoding='utf8') as fw:
        with open(filepath_predict, 'r', encoding='utf8') as fr:
            for row in fr:
                currRow = row.rstrip()
                if not currRow:
                    # new group start
                    trackRows = 1
                    refSent = ''
                    canSent = ''
                    score = 0.0
                    transBy = ''
                    fw.write(row)
                else:
                    if trackRows == 2:
                        refSent = currRow
                        fw.write(row)
                    elif trackRows == 3:
                        canSent = currRow
                        fw.write(row)
                    elif trackRows == 4:
                        score = float(currRow)
                        fw.write(row)
                    elif trackRows == 5:
                        list_ref.append(currRow)
                        prediction = predictSent(refSent, canSent, score, classifier, word_features)
                        list_predict.append(prediction)
                        fw.write(prediction + '\n')
                    elif trackRows == 1:
                        # chinese sentence, don't care for it
                        trackRows += 1
                        fw.write(row)
                        continue
                    trackRows += 1
    return list_ref, list_predict

def evaluateAccuracy(reference, prediction):
    tph = 0
    fph = 0
    fnh = 0
    tpm = 0
    fpm = 0
    fnm = 0
    total = len(reference)
    correct = 0
    accr = 0.0
    prech = 0.0
    recallh = 0.0
    precm = 0.0
    recallm = 0.0
    for i in range(total):
        if reference[i] == prediction[i]:
            correct += 1
        if reference[i] == 'H' and prediction[i] == 'H':
            tph += 1
        elif reference[i] == 'M' and prediction[i] == 'M':
            tpm += 1
        elif reference[i] == 'H' and prediction[i] == 'M':
            fnh += 1
            fpm += 1
        elif reference[i] == 'M' and prediction[i] == 'H':
            fnm += 1
            fph += 1
    prech = tph / (tph + fph)
    recallh = tph / (tph + fnh)
    precm = tpm / (tpm + fpm)
    recallm = tpm / (tpm + fnm)
    f1h = 2.0 * (prech * recallh) / (prech + recallh)
    f1m = 2.0 * (precm * recallm) / (precm + recallm)
    f1 = (f1h + f1m) / 2.0
    accr = correct/total
    print('accuracy:')
    print(accr)
    print('F1 score:')
    print(f1)
    print('correct / total')
    print(str(correct) + '/' + str(total))
    return accr


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])