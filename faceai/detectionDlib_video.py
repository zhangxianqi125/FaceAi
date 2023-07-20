# coding=utf-8
# 图片检测 - Dlib版本
import cv2
import dlib

# 打开摄像头并获取视频流
cap = cv2.VideoCapture(0)

# 人脸分类器
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor(
    "..\\model\\shape_predictor_68_face_landmarks.dat"
)
while True:
    # 从视频流中读取帧
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)
    for face in dets:
        # 在图片中标注人脸，并显示
        # left = face.left()
        # top = face.top()
        # right = face.right()
        # bottom = face.bottom()
        # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        # cv2.imshow("image", img)

        shape = predictor(img, face)  # 寻找人脸的68个标定点
        # 遍历所有点，打印出其坐标，并圈出来
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 1, (0, 255, 0), 2)

    cv2.imshow("image", img)

    # 检测用户是否按下了某个键来退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
