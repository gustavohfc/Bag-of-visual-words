import cv2
import os
import glob
import pickle
import numpy as np
from sklearn.cluster import KMeans


def getFilesAndDescriptors(path):
    infoList = []
    classCount = 0
    surf = cv2.xfeatures2d.SURF_create(400)

    for each in glob.glob(path + "*"):
        word = each.split("/")[-1]
        print(" #### Reading image category ", word, " ##### ")

        classCount += 1

        for imagefile in glob.glob(path+word+"/*"):
            im = cv2.imread(imagefile, 0)
            desc = features(surf, im)

            if desc is not None:
                infoList.append((word, desc[:100]))

    return infoList, classCount


def extractTuple(tupleList, position):
    ret = [x[position] for x in tupleList]
    return ret


def features(surf, image):
    _, descriptors = surf.detectAndCompute(image, None)
    return descriptors


def generateCodeBook(descList, classCount):
    return KMeans(n_clusters=classCount * 50, random_state=0).fit(descList)


def makeHistogram(predict, n_clusters):
    hist = np.zeros(n_clusters)

    for cluster in predict:
        hist[cluster] += 1

    return hist


def calculateHistograms(classList, descList, codeBook):
    histograms = {}

    for i, category in enumerate(classList):
        predict = codeBook.predict(descList[i])

        imHist = makeHistogram(predict, codeBook.n_clusters)

        if category in histograms:
            histograms[category] += imHist
        else:
            histograms[category] = imHist

    # Normalize histograms
    for category in histograms:
        hist_sum = sum(histograms[category])
        histograms[category] /= hist_sum

    return histograms


def classifyImage(imDesc, codeBook, histograms):
    imPredict = codeBook.predict(imDesc)
    imHist = makeHistogram(imPredict, codeBook.n_clusters)
    greaterSum = 0

    for category in histograms:
        categorySum = sum(imHist * histograms[category])

        if greaterSum < categorySum:
            imCategory = category
            greaterSum = categorySum

    return imCategory


def test(codeBook, histograms):
    testInfoList, _ = getFilesAndDescriptors("Images/Test/")

    testClassList = extractTuple(testInfoList, 0)
    testDescList = np.array(extractTuple(testInfoList, 1))

    correct_total = 0

    correct_per_category = {}
    images_per_category = {}

    for imDesc, imClass in zip(testDescList, testClassList):
        category = classifyImage(imDesc, codeBook, histograms)
        
        if category not in images_per_category:
            correct_per_category[category] = 0
            images_per_category[category] = 0

        if imClass == category:
            correct_total += 1
            correct_per_category[category] += 1

        images_per_category[category] += 1

    print('Acertos total:\t', str(100 * correct_total / len(testClassList)), ' %')

    for category in images_per_category:
        print(category,' : ', str(100 * correct_per_category[category] / images_per_category[category]), ' %')


def main():
    trainInfoList, classCount = getFilesAndDescriptors("Images/Train/")

    trainClassList = extractTuple(trainInfoList, 0)
    trainDescList = np.array(extractTuple(trainInfoList, 1))

    if os.path.exists('codeBook.pkl'):
        with open('codeBook.pkl', 'rb') as inputFile:
            codeBook = pickle.load(inputFile)
    else:
        codeBook = generateCodeBook(np.vstack(trainDescList), classCount)

        with open('codeBook.pkl', 'wb') as outputFile:
            pickle.dump(codeBook, outputFile, pickle.HIGHEST_PROTOCOL)


    if os.path.exists('histograms.pkl'):
        with open('histograms.pkl', 'rb') as inputFile:
            histograms = pickle.load(inputFile)
    else:
        histograms = calculateHistograms(trainClassList, trainDescList, codeBook)

        with open('histograms.pkl', 'wb') as outputFile:
            pickle.dump(histograms, outputFile, pickle.HIGHEST_PROTOCOL)

    test(codeBook, histograms)


if __name__ == '__main__':
    main()