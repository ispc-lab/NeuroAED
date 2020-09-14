import numpy as np
import math

eventfile = '/home/lpg/PycharmProjects/AED/anomaly_dataset/campus/events/nor/N1.txt'
# offile = '/home/lpg/PycharmProjects/AED/anomaly_dataset/OFfix/N1fix.txt'

# def ACuboid(starttime = -1, poithr = 200):

def ACuboid(starttime = -1):
    eventmat = np.loadtxt(eventfile)
    startevent = 0
    if starttime == -1:
        print("starttime is defaulted to be ", eventmat[0][0])
        starttime = eventmat[0][0]

    else:
        for i in range(len(eventmat)):
            if eventmat[i][0] < starttime and eventmat[i+1][0] >= starttime:
                startevent = i + 1
                break


    # k = 0
    f = []
    x = []
    y = []
    count = []
    # while startevent < len(eventmat) - poithr and eventmat[-1][0] - starttime > 100000:
    while eventmat[-1][0] - starttime > 100000:
        cuboid = np.zeros((10, 10), dtype=np.int)
        for i in range(startevent, len(eventmat)):
            if eventmat[i][0] <=  starttime + 100000:
                if eventmat[i][1] < 340:
                    cuboid[int(eventmat[i][2]/26)][int(eventmat[i][1]/34)] += 1
                else:
                    cuboid[int(eventmat[i][2] /26)][9] += 1
            else:
                startevent = i
                starttime += 100000
                break

        for i in range(10):
            for j in range(10):
                # if cuboid[i][j] >= poithr:
                y.append(i)
                x.append(j)
                f.append(starttime - 100000)
                count.append(cuboid[i][j])
        # print(starttime, startevent)
    # actcubmat = np.vstack((f, x, y)).T
    return f, x, y, count

if __name__ == "__main__":
    f, x, y, count = ACuboid()       # 确定activate cuboid
    ACuboid = np.vstack((f, x, y, count)).T
    np.savetxt('/home/lpg/桌面/test/countuprun.txt', ACuboid, fmt=("%d %d %d %d"))
    