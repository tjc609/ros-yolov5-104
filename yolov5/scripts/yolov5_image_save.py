# 保存视频到指定路径，并且文件名包含时间戳，每次保存的视频文件名不同
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
from datetime import datetime

class VideoSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        self.save_path = "/home/chengzai/catkin_ws_yolo/src/yolov5/yolov5-master/data/videos"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # 动态生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = self.generate_filename(timestamp)
        
        # VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
        self.out = None
        self.frame_width = 640  # Default frame width, adjust as needed
        self.frame_height = 480  # Default frame height, adjust as needed
        self.fps = 30  # Default frames per second, adjust as needed

    def generate_filename(self, timestamp):
        index = 1
        while True:
            filename = os.path.join(self.save_path, f"{timestamp}_{index}.avi")
            if not os.path.exists(filename):
                return filename
            index += 1

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Initialize the VideoWriter object once we receive the first frame
        if self.out is None:
            self.out = cv2.VideoWriter(self.video_filename, self.fourcc, self.fps, 
                                       (self.frame_width, self.frame_height))

        # Write the frame into the file
        self.out.write(cv_image)

    def cleanup(self):
        if self.out is not None:
            self.out.release()
            rospy.loginfo(f"Video saved as {self.video_filename}")
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