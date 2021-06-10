# USAGE
# python mask_rcnn_segmentation.py --input ../example_videos/dog_park.mp4 --output ../output_videos/mask_rcnn_dog_park.avi --display 0 --mask-rcnn mask-rcnn-coco
# python mask_rcnn_segmentation.py --input ../example_videos/dog_park.mp4 --output ../output_videos/mask_rcnn_dog_park.avi --display 0 --mask-rcnn mask-rcnn-coco --use-gpu 1
# python mask_rcnn_segmentation.py --display 1 --mask-rcnn mask-rcnn-coco

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import os

# построить анализ аргумента и проанализировать аргументы
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", type=str, default="", help="путь к (необязательно) входному видеофайлу")
# ap.add_argument("-o", "--output", type=str, default="", help="путь к (необязательно) выходному видеофайлу")
# ap.add_argument("-d", "--display", type=int, default=1, help="должен ли отображаться кадр вывода")
# ap.add_argument("-m", "--mask-rcnn", required=True, help="базовый путь к каталогу mask-rcnn")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="минимальная вероятность отфильтровать слабые обнаружения")
# ap.add_argument("-t", "--threshold", type=float, default=0.3, help="минимальный порог для пиксельной сегментации маски")
# ap.add_argument("-u", "--use-gpu", type=bool, default=0,
#                 help="логическое значение, указывающее, следует ли использовать CUDA GPU")
# args = vars(ap.parse_args())

mask_rcnn = "mask-rcnn-coco"
# загружаем метки классов COCO, на которых была обучена наша маска R-CNN
labelsPath = os.path.sep.join([mask_rcnn, "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# инициализировать список цветов для представления каждой возможной метки класса
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# получить пути к весам Mask R-CNN и конфигурации модели
weightsPath = os.path.sep.join([mask_rcnn, "frozen_inference_graph.pb"])
configPath = os.path.sep.join([mask_rcnn, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# загрузите нашу маску R-CNN, обученную на наборе данных COCO (90 классов) с диска
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# проверь, собираемся ли мы использовать GPU
use_gpu = 0
if use_gpu:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# инициализировать видеопоток и указатель для вывода видеофайла, затем запустить таймер FPS
print("[INFO] accessing video stream...")
input_data = ""
vs = cv2.VideoCapture(input_data if input_data else 0)
writer = None
fps = FPS().start()

# перебирать кадры из потока видеофайлов
while True:
    # прочитать следующий кадр из файла
    (grabbed, frame) = vs.read()
    # если кадр не был схвачен, значит, мы достигли конца потока
    if not grabbed:
        break

    # создать blob из входного кадра, а затем выполнить прямой проход Mask R-CNN, дав нам (1) координаты ограничивающего
    # прямоугольника объектов на изображении вместе с (2) пиксельной сегментацией для каждого конкретного объекта
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

    # перебрать количество обнаруженных объектов
    for i in range(0, boxes.shape[2]):
        # извлеките идентификатор класса обнаружения вместе с достоверностью (то есть вероятностью),
        # связанной с прогнозом
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        confidence_minimum_value = 0.5
        # отфильтровать слабые прогнозы, гарантируя, что обнаруженная вероятность больше минимальной вероятности
        if confidence > confidence_minimum_value:
            # масштабируйте координаты ограничивающего прямоугольника обратно относительно размера кадра, а затем
            # вычислите ширину и высоту ограничивающего прямоугольника
            (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # извлеките пиксельную сегментацию для объекта, измените размер маски так, чтобы она соответствовала
            # размерам ограничивающей рамки, а затем, наконец, порог для создания * двоичной * маски
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
            threshold = 0.3
            mask = (mask > threshold)

            # извлекать ROI изображения, но * только * извлекать замаскированную область ROI
            roi = frame[startY:endY, startX:endX][mask]

            # возьмите цвет, используемый для визуализации этого конкретного класса,
            # затем создайте прозрачное наложение, смешав цвет с ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # сохранить смешанный ROI в исходном кадре
            frame[startY:endY, startX:endX][mask] = blended

            # нарисуйте ограничивающую рамку экземпляра на фрейме
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # нарисуйте прогнозируемую метку и связанную с ней вероятность сегментации экземпляра на кадре
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # проверьте, должен ли выходной кадр отображаться на вашем экране
    display = 1
    if display > 0:
        # показать выходной кадр
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # если была нажата клавиша `q`, выйти из цикла
        if key == ord("q"):
            break

    # если был указан путь к выходному видеофайлу, а средство записи видео не было инициализировано, сделайте это сейчас
    output = ""
    if output != "" and writer is None:
        # инициализировать наш видео писатель
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # если видеомагнитофон None, записать кадр в выходной видеофайл
    if writer is not None:
        writer.write(frame)

    # обновить счетчик FPS
    fps.update()

# остановить таймер и отобразить информацию о FPS
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
