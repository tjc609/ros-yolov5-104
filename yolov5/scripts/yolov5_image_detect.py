#!/usr/bin/env python  #指定Python解释器,这个特别关键，如果有虚拟环境报错的问题，可以使用这个方法
# 实时显示摄像头的RGB图像，并使用YOLOv5检测物体,显示物体的2D中心坐标
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import torch  # 添加对torch的导入


class VideoSaver:
    def __init__(self):
        rospy.loginfo("Initializing VideoSaver...")
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        # 初始化YOLOv5模型
        yolov5_path = '/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master'
        weight_path = os.path.join(yolov5_path, 'yolov5s.pt')  # 使用预训练模型
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')
        self.model.conf = 0.5  # 设置置信度阈值
        rospy.loginfo("YOLOv5 model initialized.")

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
            class_name = self.model.names[int(cls)]  # 获取类别名称
            rospy.loginfo(f"Detected object: Class={class_name}, Confidence={conf:.2f}, Center=({center_x}, {center_y})")

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