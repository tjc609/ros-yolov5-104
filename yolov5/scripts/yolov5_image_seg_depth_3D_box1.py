#!/usr/bin/env python
# 实时显示摄像头的RGB图像，并使用YOLOv5分割物体, 显示物体的3D边界框和中心点，
# 过滤深度图像噪声，显示深度最大值，最小值和平均值，显示物体的3D坐标，长度，高度和厚度

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import torch
import numpy as np

from scipy import ndimage

import sys
sys.path.append('/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master')
from ultralytics.utils.plotting import Annotator, colors
from utils.general import (
    non_max_suppression,
    scale_boxes,
)
from utils.segment.general import process_mask

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class VideoSaver:
    def __init__(self):
        rospy.loginfo("Initializing VideoSaver...")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        
        self.depth_image = None
        self.filtered_depth = None

        yolov5_path = '/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master'
        weight_path = os.path.join(yolov5_path, 'yolov5s-seg.pt')
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        self.model.conf = 0.7 # 置信度阈值
        self.model.iou = 0.45 # IOU阈值
        
        # 更新后的相机内参矩阵
        self.fx = 521.65727  # 焦距在x方向上的像素值
        self.fy = 515.23771  # 焦距在y方向上的像素值
        self.cx = 292.25207  # 主点x坐标
        self.cy = 217.50548  # 主点y坐标
        
        rospy.loginfo("YOLOv5 model initialized.")
    
    def advanced_depth_filter(self, depth_image, max_depth=10000):
        """
        使用高级方法过滤深度图像噪声
        :param depth_image: 原始深度图像
        :param max_depth: 最大有效深度值（单位：毫米）
        :return: 过滤后的深度图像
        """
        # 步骤1: 移除无效值
        depth_image = np.where(depth_image > max_depth, 0, depth_image)
        depth_image = np.where(np.isnan(depth_image), 0, depth_image)
        
        # 步骤2: 双边滤波
        depth_image_float = depth_image.astype(np.float32)
        filtered = cv2.bilateralFilter(depth_image_float, d=5, sigmaColor=50, sigmaSpace=50)
        
        # 步骤3: 中值滤波
        filtered = cv2.medianBlur(filtered.astype(np.uint16), 5)
        
        # 步骤4: 形态学闭操作填充小孔
        kernel = np.ones((5,5), np.uint8)
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        # 步骤5: 统计离群值去除
        mask = filtered > 0
        mean = np.mean(filtered[mask])
        std = np.std(filtered[mask])
        filtered = np.where((filtered > mean - 2*std) & (filtered < mean + 2*std), filtered, 0)
        
        # 步骤6: 小连通域去除
        labeled_array, num_features = ndimage.label(filtered > 0)
        sizes = ndimage.sum(filtered > 0, labeled_array, range(1, num_features + 1))
        mask_size = sizes < 100  # 可以调整这个阈值
        remove_pixel = mask_size[labeled_array - 1]
        filtered[remove_pixel] = 0
        
        return filtered.astype(np.uint16)

    
    def get_bbox_and_center_from_mask(self, mask):
        # 确保掩码是二值图像
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # 找到掩码的非零像素
        non_zero_coords = cv2.findNonZero(mask)
        
        if non_zero_coords is None or len(non_zero_coords) == 0:
            return None, None

        # 计算中心点
        center_x = np.mean(non_zero_coords[:, 0, 0])
        center_y = np.mean(non_zero_coords[:, 0, 1])
        center = (int(center_x), int(center_y))

        # 计算边界框
        x, y, w, h = cv2.boundingRect(non_zero_coords)
        
        return (x, y, x+w, y+h), center
    

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            if self.depth_image is not None:
                self.filtered_depth = self.advanced_depth_filter(self.depth_image)
            else:
                rospy.logwarn("Received empty depth image")
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
        
        if self.filtered_depth is None:
            rospy.logwarn("Filtered depth image is not available.")
            return
        
        im = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        im = np.transpose(im, (2, 0, 1))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]
        with torch.no_grad():
            pred, proto = self.model(im)[:2]
        pred = non_max_suppression(pred, self.model.conf, self.model.iou, classes=None, agnostic=False, max_det=1000, nm=32)
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], cv_image.shape).round()
                masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
                annotator = Annotator(cv_image, line_width=3, example=str(self.model.names))
                im_gpu = im[0] if im.dim() == 4 else im
                annotator.masks(masks, colors=[colors(x, True) for x in det[:, 5]], im_gpu=im_gpu)
                class_names = ["person", "bicycle", "car", "motorcycle",
                                "bottle", "bus", "train", "truck", "boat",
                                "traffic light", "fire hydrant", "stop sign", 
                                "parking meter", "bench", "bird", "cat",
                                "dog","horse", "sheep", "cow", "elephant", 
                                "bear","zebra", "giraffe", "backpack", "umbrella",
                                "handbag", "tie", "suitcase", "frisbee",
                                "skis", "snowboard", "sports ball", "kite", 
                                "baseball bat", "baseball glove", "skateboard", 
                                "surfboard", "tennis racket", "bottle", 
                                "wine glass", "cup", "fork", "knife", "spoon", 
                                "bowl", "banana", "apple", "sandwich", "orange", 
                                "broccoli", "carrot", "hot dog", "pizza", "donut", 
                                "cake", "chair", "couch", "potted plant", "bed", 
                                "dining table", "toilet", "tv", "laptop", "mouse", 
                                "remote", "keyboard", "cell phone", "microwave", 
                                "oven", "toaster", "sink", "refrigerator", "book", 
                                "clock", "vase", "scissors", "teddy bear", "hair drier", 
                                "toothbrush"]
                
                for mask, conf, cls in zip(masks, det[:, 4], det[:, 5]):
                    if not isinstance(mask, np.ndarray):
                        if mask.is_cuda:
                            mask = mask.cpu().numpy()
                        else:
                            mask = mask.numpy()
                    
                    # 通过掩码获取边界框和中心点
                    bbox, center = self.get_bbox_and_center_from_mask(mask)
                    if bbox is None or center is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    
                    # 获取物体区域的深度值
                    object_depth = self.filtered_depth[y1:y2, x1:x2]
                    valid_depths = object_depth[object_depth > 0]

                    if len(valid_depths) == 0:
                        continue  # 跳过没有有效深度信息的物体
            
                    min_depth = np.min(valid_depths) / 1000.0  # 转换为米
                    max_depth = np.max(valid_depths) / 1000.0  # 转换为米
                    avg_depth = np.mean(valid_depths) / 1000.0  # 转换为米

                    # 计算avg_depth和min_depth的差值
                    depth_difference = avg_depth - min_depth

                    # 使用最小深度值进行后续计算
                    depth_value_m = min_depth
                    
                    # 将2D图像坐标转换为3D世界坐标
                    X = (center[0] - self.cx) * depth_value_m / self.fx
                    Y = (center[1] - self.cy) * depth_value_m / self.fy
                    Z = depth_value_m
                    
                    # 计算边界框的长度和高度（以米为单位）
                    width_pixels = x2 - x1
                    height_pixels = y2 - y1
                    width_meters = width_pixels * Z / self.fx
                    height_meters = height_pixels * Z / self.fy

                    # 显示3D世界坐标和边界框
                    label = f'{class_names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 计算厚度（深度）
                    depth_meters = depth_difference * 2

                    # 更新显示边界框的长度、高度和厚度的代码
                    size_text = f'Size: L:{width_meters:.2f}m x H:{height_meters:.2f}m x D:{depth_meters:.2f}m'
                    cv2.putText(cv_image, size_text, (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 显示边界框四个角的2D平面坐标（红色）
                    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    for i, corner in enumerate(corners):
                        coord_text = f'({corner[0]},{corner[1]})'
                        if i == 0:  # 左上角
                            cv2.putText(cv_image, coord_text, (corner[0] - 30, corner[1] + 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        elif i == 1:  # 右上角
                            cv2.putText(cv_image, coord_text, (corner[0] + 30, corner[1] + 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        elif i == 2:  # 右下角
                            cv2.putText(cv_image, coord_text, (corner[0] + 30, corner[1] - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        else:  # 左下角
                            cv2.putText(cv_image, coord_text, (corner[0] - 30, corner[1] - 15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # 显示中心的2D坐标（紫色）
                    center_text = f'Center: ({center[0]},{center[1]})'
                    cv2.putText(cv_image, center_text, (center[0] - 60, center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    
                    label_x = f'X={X:.3f}'
                    label_y = f'Y={Y:.3f}'
                    label_z = f'Z={Z:.3f}'
                    cv2.putText(cv_image, label_x, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(cv_image, label_y, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(cv_image, label_z, (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.circle(cv_image, center, 3, (0, 0, 255), -1)
                    
                    # 显示深度信息
                    depth_text = f'Depth: Min={min_depth:.3f}m, Max={max_depth:.3f}m, Avg={avg_depth:.3f}m'
                    cv2.putText(cv_image, depth_text, (x1, y2 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                   # 更新显示深度差值的代码
                    diff_text = f'Depth Difference: {depth_difference:.3f}m, Depth: {depth_meters:.3f}m'
                    cv2.putText(cv_image, diff_text, (x1, y2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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