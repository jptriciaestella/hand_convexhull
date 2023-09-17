import cv2
import numpy as np
import math

# OPEN CV VERSION 4

video_capture = cv2.VideoCapture(0)

while(video_capture.isOpened()):
    # read video capture: return bool success read dan img
    ret, img = video_capture.read()

    # membaca instance tangan pada batas window yang sudah diberikan
    cv2.rectangle(img, (300,300), (100,100), (0,255,0),0)
    crop_img = img[100:300, 100:300]

    # mengubah gambar menjadi grayscale
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # menggunakan gaussian blur untuk mengurangi noise pada gambar
    blurred_img = cv2.GaussianBlur(gray_img, (35,35), 0)

    # thresholding: Otsu's Binarization method
    # referensi Otsu's Binarization: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html 
    _, thresh1 = cv2.threshold(blurred_img, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # mencari contours
    # referensi: https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
    contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_NONE)

    # mencari contour dengan area terbesar
    contour = max(contours, key = lambda x: cv2.contourArea(x))

    # mencari convex hull
    hull = cv2.convexHull(contour)

    # menggambar contour dan convex hull
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing, [contour], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # mencari convex hull
    hull = cv2.convexHull(contour, returnPoints=False)

    # Mencari defects
    defects = cv2.convexityDefects(contour, hull)
    count_defects = 0

    # Menggunakan aturan cosinus untuk menghitung sudut tiap defects
    # Menyimpan defects dengan sudut lebih kecil dari 90 derajat
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        # Mencari tiga titik pada contour
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Mencari panjang setiap sisi pada segitiga
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # Menghitung sudut berdasarkan panjang sisi segitiga menggunakan aturan cos
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # Menambahkan sudut yang valid kedalam array
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)

        # menggambar garis dari convex point
        cv2.line(crop_img,start, end, [0,255,0], 2)

    # Menampilkan hasil prediksi berapa jari yang sedang ditampilkan
    font = cv2.FONT_HERSHEY_PLAIN
    if count_defects == 1:
        cv2.putText(img,"2", (50, 50), font, 2, 2)
    elif count_defects == 2:
        cv2.putText(img, "3", (5, 50), font, 2, 2)
    elif count_defects == 3:
        cv2.putText(img,"4", (50, 50), font, 2, 2)
    elif count_defects == 4:
        cv2.putText(img,"5", (50, 50), font, 2, 2)
    else:
        cv2.putText(img,"Cannot read", (50, 50), font, 2, 2)

    # Menampilkan gambar video capture dan contours
    cv2.imshow('Gesture (Press x to quit)', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours (Press x to quit)', all_img)

    # Menutup aplikasi ketika menekan x
    if ((cv2.waitKey(1) & 0xFF) == ord('x')):
        break

video_capture.release()
cv2.destroyAllWindows()