import cv2
import glob
import numpy


def getFiles(path):
    imList = {}
    count = 0
    surf = cv2.xfeatures2d.SURF_create(400)

    for each in glob.glob(path + "*"):
        word = each.split("/")[-1]
        print(" #### Reading image category ", word, " ##### ")
        imList[word] = []

        for imagefile in glob.glob(path+word+"/*"):
            im = cv2.imread(imagefile, 0)
            desc = features(surf, im)

            print("Reading file " + imagefile + " - Desc: " + str(len(desc)))

            imList[word].append((im, desc))
            count +=1

    return imList, count


def features(surf, image):
    _, descriptors = surf.detectAndCompute(image, None)
    return descriptors


def main():
    imTrainList, Traincount = getFiles("Images/Train/")


if __name__ == '__main__':
    main()