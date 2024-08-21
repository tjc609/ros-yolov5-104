# !/usr/bin/env python
# 实时显示摄像头的RGB图像，并使用YOLOv5分割物体,显示分割物体的2D中心坐标

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import torch  # 添加对torch的导入
import numpy as np  # 添加对numpy的导入

import sys
sys.path.append('/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master')
from ultralytics.utils.plotting import Annotator, colors
from utils.general import (
    cv2,
    non_max_suppression,
    scale_boxes,
)
from utils.segment.general import process_mask

class VideoSaver:
    def __init__(self):
        rospy.loginfo("Initializing VideoSaver...")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

        # 初始化YOLOv5模型
        yolov5_path = '/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master'
        weight_path = os.path.join(yolov5_path, 'yolov5s-seg.pt')  # 使用预训练模型
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        self.model.conf = 0.7  # 设置置信度阈值

        rospy.loginfo("YOLOv5 model initialized.")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        if not hasattr(self, 'model'):
            rospy.logerr("Model attribute is missing.")
            return

        im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        im = np.transpose(im, (2, 0, 1))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        with torch.no_grad():
            pred, proto = self.model(im)[:2]

        pred = non_max_suppression(pred, self.model.conf, 0.45, classes=None, agnostic=False, max_det=1000, nm=32)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], cv_image.shape).round()
                masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                annotator = Annotator(cv_image, line_width=3, example=str(self.model.names))
                im_gpu = im[0] if im.dim() == 4 else im
                annotator.masks(masks, colors=[colors(x, True) for x in det[:, 5]], im_gpu=im_gpu)

                # 假设类别名称存储在一个列表中
                class_names = ["person", "bicycle", "car", "motorcycle", "bottle", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

                for mask, conf, cls in zip(masks, det[:, 4], det[:, 5]):
                    if not isinstance(mask, np.ndarray):
                        if mask.is_cuda:
                            mask = mask.cpu().numpy()
                        else:
                            mask = mask.numpy()
                    non_zero_coords = cv2.findNonZero(mask)
                    if non_zero_coords is not None:
                        center_x = np.mean(non_zero_coords[:, 0, 0])
                        center_y = np.mean(non_zero_coords[:, 0, 1])
                        center = (int(center_x), int(center_y))
                        label = f'{class_names[int(cls)]} {conf:.2f} ({center[0]}, {center[1]})'
                        cv2.putText(cv_image, label, (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # 输出中心点坐标
                        print(f'类别: {class_names[int(cls)]}, 置信度: {conf:.2f}, 中心点: {center}')
                        # 在图像上显示中心点坐标
                        cv2.circle(cv_image, center, 3, (0, 0, 255), -1)
                    else:
                        print("掩码中没有非零像素")

        cv2.imshow("YOLOv5 Segmentation", cv_image)
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()

def main():
    rospy.init_node('video_saver', anonymous=True)
    video_saver = VideoSaver()
    rospy.on_shutdown(video_saver.cleanup)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()