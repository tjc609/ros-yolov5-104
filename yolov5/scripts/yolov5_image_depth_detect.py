#!/usr/bin/env python
# 实时显示摄像头的RGB图像，并使用YOLOv5检测物体，同时获取物体中心的深度信息
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import torch

class VideoSaver:
    def __init__(self):
        rospy.loginfo("Initializing VideoSaver...")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        
        # 初始化YOLOv5模型
        yolov5_path = '/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master'
        weight_path = os.path.join(yolov5_path, 'yolov5s.pt')  # 使用预训练模型
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        self.model.conf = 0.5  # 设置置信度阈值
        rospy.loginfo("YOLOv5 model initialized.")
        
        self.depth_image = None

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    def get_distance(self, x, y):
        if self.depth_image is not None:
            distance = self.depth_image[y, x] * 0.001  # 转换为米
            return distance
        else:
            rospy.logwarn("Depth image not received yet")
            return None

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # YOLOv5检测
        if not hasattr(self, 'model'):
            rospy.logerr("Model attribute is missing.")
            return

        results = self.model(cv_image)
        detected_img = results.render()[0]  # 获取带有检测框的图像

        # 遍历检测结果并在终端显示信息
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            distance = self.get_distance(center_x, center_y)
            class_name = self.model.names[int(cls)]  # 获取类别名称
            if distance is not None:
                rospy.loginfo(f"Detected object: Class={class_name}, Confidence={conf:.2f}, Center=({center_x}, {center_y}), Distance={distance:.2f}m")
            else:
                rospy.logwarn(f"Failed to get distance for object at Center=({center_x}, {center_y})")

        # 实时显示检测结果
        cv2.imshow("Detected Video", detected_img)
        cv2.waitKey(1)  # 1ms 延迟以允许显示图像

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