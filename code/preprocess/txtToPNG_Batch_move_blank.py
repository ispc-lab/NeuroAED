import numpy as np
import cv2
import os


def loadTxt(file):
    count = 0
    ts =[]
    x =[]
    y =[]
    pol =[]
    with open(file) as f:
        for line in f:
            items = line.split()
            ts.insert(count, float(items[0]))
            x.insert(count, int(items[1]))
            y.insert(count, int(items[2]))
            pol.insert(count, int(items[3]))
            # print(items)

            count = count +1

    return ts, x, y, pol


def move_blank_1(txtfolder_path, imgfolder_path, subject_num, gesture_num):

    """
        function: accumulate data to single channel frames, then remove blank
    """
    for m in range(subject_num):
        new_img_folder = imgfolder_path + str(m+1)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)
        for n in range(gesture_num):

            # Convert txt into PNG and store them in ../frames
            # new_folder = imgfolder_path + str(i+1) + '/' + str(j+1)
            new_folder = imgfolder_path + str(m+1) + '/' + str(n+1)
            if not os.path.exists(new_folder):
                    os.makedirs(new_folder)

            txtfile = txtfolder_path + str(m+1) + '/' + str(n+1) + '.txt'
            imgfile_path = imgfolder_path + str(m+1) + '/' + str(n+1) + '/'

            t, x, y, pol = loadTxt(txtfile)
            x[:] = [int(a-1) for a in x]
            y[:] = [int(a-1) for a in y]

            img = np.zeros((640, 768), dtype=np.uint8)

            idx = 0
            start_idx = 0
            startTime = 0
            endTime = 0
            stepTime = 10000/0.08
            imgCount = 1

            while startTime < t[-1]:
                endTime = startTime + stepTime
                while t[idx] < endTime and idx < len(t)-1:
                    idx = idx + 1
                # reshape关键在后面的列1，前面的-1没有意义
                data_x = np.array(x[start_idx:idx]).reshape((-1, 1))
                data_y = np.array(y[start_idx:idx]).reshape((-1, 1))
                # data_t = np.array(t[start_idx:idx]).reshape((-1, 1))
                # 以stack列的方式合并俩数组
                data = np.column_stack((data_x, data_y))
                data_filter = data

                for i in range(0, data_filter.shape[0]):
                    img[data_filter[i][1] - 1][data_filter[i][0] - 1] = 255

                start_idx = idx
                startTime = t[idx]
                # 计算值为非黑的点...
                print(sum(img[img > 0]))
                # TO DO: 滤掉有条纹的
                if (sum(img[img > 0]) > 1000000) & (sum(img[img > 0]) < 50000000):
                    # img = cv2.flip(img, 0)
                    # cv2.imshow('dvs', img)
                    # cv2.waitKey(5)
                    imgFullFile = imgfile_path + ('%05d' % imgCount) + '.png'
                    cv2.imwrite(imgFullFile, img)
                    imgCount = imgCount + 1
                # 如果不加就会变成motion history
                img[:] = 0
                # start_idx = idx
                #
                # # img = cv2.flip(img, 0)
                # cv2.imshow('dvs', img)
                # cv2.waitKey(5)
                # imgFullFile = imgfile_path+ ('%05d' % imgCount) + '.png'
                # cv2.imwrite(imgFullFile, img)
                #
                # img[:] = 0
                # startTime = t[idx]
                # imgCount = imgCount + 1
                # #

                print('finished convert TxtToPNG for {}'.format(txtfile))


def move_blank_2(txtfolder_path, imgfolder_path, subject_num, gesture_num):

    """
        function: accumulate data to 3 channel frames, then remove blank
    """
    for m in range(subject_num):
        new_img_folder = imgfolder_path + str(m + 1)
        if not os.path.exists(new_img_folder):
            os.makedirs(new_img_folder)
        for n in range(gesture_num):

            # Convert txt into PNG and store them in ../frames
            # new_folder = imgfolder_path + str(i+1) + '/' + str(j+1)
            new_folder = imgfolder_path + str(m + 1) + '/' + str(n + 1)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

            txtfile = txtfolder_path + str(m + 1) + '/' + str(n + 1) + '.txt'
            imgfile_path = imgfolder_path + str(m + 1) + '/' + str(n + 1) + '/'

            t, x, y, pol = loadTxt(txtfile)
            x[:] = [int(a - 1) for a in x]
            y[:] = [int(a - 1) for a in y]

            img = np.zeros((640, 768, 3), dtype=np.uint8)

            idx = 0
            start_idx = 0
            startTime = 0
            endTime = 0
            stepTime = 10000 / 0.08
            imgCount = 1

            while startTime < t[-1]:
                endTime = startTime + stepTime
                while t[idx] < endTime and idx < len(t) - 1:
                    idx = idx + 1
                # reshape关键在后面的列1，前面的-1没有意义
                data_x = np.array(x[start_idx:idx]).reshape((-1, 1))
                data_y = np.array(y[start_idx:idx]).reshape((-1, 1))
                data_t = np.array(t[start_idx:idx]).reshape((-1, 1))
                # 以stack列的方式合并俩数组
                data = np.column_stack((data_x, data_y, data_t))
                data_filter = data

                for i in range(0, data_filter.shape[0]):
                    # img[x-1][y-1], & multi-cue
                    img[int(data_filter[i][1] - 1)][int(data_filter[i][0] - 1)][0] += 85  # channel frequency
                    img[int(data_filter[i][1] - 1)][int(data_filter[i][0] - 1)][1] = 255  # channel NONE
                    img[int(data_filter[i][1] - 1)][int(data_filter[i][0] - 1)][2] = 255 * (
                            data_filter[i][2] - t[start_idx]) / (t[idx] - t[start_idx])  # channel time stamp

                start_idx = idx
                startTime = t[idx]
                # 计算值为非黑的点...
                print(sum(img[img > 0]))
                # TO DO: 滤掉有条纹的
                if (sum(img[img > 0]) > 1000000) & (sum(img[img > 0]) < 50000000):
                    # img = cv2.flip(img, 0)
                    # cv2.imshow('dvs', img)
                    # cv2.waitKey(5)
                    imgFullFile = imgfile_path + ('%05d' % imgCount) + '.png'
                    cv2.imwrite(imgFullFile, img)
                    imgCount = imgCount + 1
                # 如果不加就会变成motion history
                img[:] = 0
                # start_idx = idx
                #
                # # img = cv2.flip(img, 0)
                # cv2.imshow('dvs', img)
                # cv2.waitKey(5)
                # imgFullFile = imgfile_path+ ('%05d' % imgCount) + '.png'
                # cv2.imwrite(imgFullFile, img)
                #
                # img[:] = 0
                # startTime = t[idx]
                # imgCount = imgCount + 1
                # #

                print('finished convert TxtToPNG for {}'.format(txtfile))


if __name__ == "__main__":

    current_path = 'F:/Advanced_Research_Guang/KeyGesture/raw_gesture/'
    txtfolder_path = 'F:/Advanced_Research_Guang/KeyGesture/raw_gesture/txt_list/'
    imgfolder_path = 'F:/Advanced_Research_Guang/KeyGesture/raw_gesture/img_remove_blank_single_ch/'

    subject_num = 13
    gesture_num = 5
    move_blank_1(txtfolder_path, imgfolder_path, subject_num, gesture_num)
