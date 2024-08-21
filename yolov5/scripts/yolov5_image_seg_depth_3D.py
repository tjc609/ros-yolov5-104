#!/home/chengzai/anaconda3/envs/yolov5/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import torch
import numpy as np

import sys
sys.path.append('/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master')
from ultralytics.utils.plotting import Annotator, colors
from utils.general import (
    non_max_suppression,
    scale_boxes,
)
from utils.segment.general import process_mask



class VideoSaver:
    def __init__(self):
        rospy.loginfo("Initializing VideoSaver...")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)

        self.depth_image = None

        yolov5_path = '/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master'
        weight_path = os.path.join(yolov5_path, 'yolov5s-seg.pt')
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        self.model.conf = 0.7

        # 相机内参矩阵
        self.fx = 579.7169799804688  # 焦距在x方向上的像素值
        self.fy = 579.7169799804688  # 焦距在y方向上的像素值
        self.cx = 320.552001953125  # 主点x坐标
        self.cy = 256.1300048828125  # 主点y坐标

        rospy.loginfo("YOLOv5 model initialized.")

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

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
                        cv2.circle(cv_image, center, 3, (0, 0, 255), -1)

                        if self.depth_image is not None:
                            depth_value_mm = self.depth_image[center[1], center[0]]
                            depth_value_m = depth_value_mm / 1000.0  # 将毫米转换为米

                            # 将2D图像坐标转换为3D世界坐标
                            X = (center[0] - self.cx) * depth_value_m / self.fx
                            Y = (center[1] - self.cy) * depth_value_m / self.fy
                            Z = depth_value_m

                            # 显示3D世界坐标
                            label_x = f'X={X:.3f}'
                            label_y = f'Y={Y:.3f}'
                            label_z = f'Z={Z:.3f}'
                            cv2.putText(cv_image, label_x, (center[0], center[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(cv_image, label_y, (center[0], center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(cv_image, label_z, (center[0], center[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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