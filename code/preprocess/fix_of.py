'''
The original optical flow file extracted by jAER may have negativate timestamps and format problems,
this procedure is to handle them.
'''

import numpy as np
import os

oforipath = '...' # Folder where original optical flow files have been stored
offixpath = '...' # Folder to store fixed optical flow files


for root, dirs, fs in os.walk(oforipath):
    for f in fs:
        ts =[]
        x = []
        y = []
        vx = []
        vy = []
        speed = []
        ofmat = open(oforipath + f, 'r')

        lt = 1
        while 1:
            lt = ofmat.readline()
            if not lt:
                break

            if lt[0] != "#":
                str1 = lt.split(",")
                str2 = str1[1].split(" ")
                if int(str2[7]) != 0:

                    x.append(int(str2[1]))
                    y.append(int(str2[2]))
                    vx.append(float(str2[4]))
                    vy.append(float(str2[5]))
                    speed.append(float(str2[6]))
                    if int(str2[0]) >= 0:
                        ts.append(int(str2[0]))
                    else:
                        ts.append(int(str2[0]) + pow(2, 32))
        newname1 = f.split("-")
        newname2 = newname1[0]
        newofmat = np.vstack((ts, x, y, vx, vy, speed)).T
        np.savetxt(offixpath + newname2 + '.txt', newofmat, fmt="%d %d %d %0.2f %0.2f %0.2f")


