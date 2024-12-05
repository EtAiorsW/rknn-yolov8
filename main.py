import cv2
import time
from rknnpool import rknnPoolExecutor
from yolov8 import yolov8

cap = cv2.VideoCapture('./test.mp4')
# cap = cv2.VideoCapture(0)
modelPath = "./models/yolov8n.rknn"
# 线程数, 增大可提高帧率
TPEs = 3
# 初始化rknn池
pool = rknnPoolExecutor(
    model=modelPath,
    TPEs=TPEs,
    func=yolov8)

# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 60帧平均帧率
    if frames % 60 == 0:
        print(f'time: {(time.time() - loopTime) * 1000 / 60:.2f} ms, fps: {60 / (time.time() - loopTime):.2f}')
        loopTime = time.time()

print("总平均帧率\t", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()
pool.release()
