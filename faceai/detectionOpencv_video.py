import cv2
import datetime

# OpenCV人脸识别分类器
classifier = cv2.CascadeClassifier(
    "..\model\haarcascade_frontalface_default.xml"
)

# 打开摄像头并获取视频流
cap = cv2.VideoCapture(0)

while True:
    # 程序开始时间
    startTime = datetime.datetime.now()

    # 从视频流中读取帧
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
            # 左眼
            cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                       color)
            #右眼
            cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                       color)
            #嘴巴
            cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                          (x + 5 * w // 8, y + 7 * h // 8), color)

    # 程序结束时间
    endTime = datetime.datetime.now()
    print((endTime - startTime))

    # 显示图像
    cv2.imshow("image", img)

    # 检测用户是否按下了某个键来退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭摄像头并释放资源
cap.release()
cv2.destroyAllWindows()
