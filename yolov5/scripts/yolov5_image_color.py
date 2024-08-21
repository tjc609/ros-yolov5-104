# !/usr/bin/env python3
# 实时显示摄像头的RGB图像
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def cam_rgb_callback(msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Display the image
    cv2.imshow("RGB Image", cv_image)
    cv2.waitKey(1)

def main():
    rospy.init_node('cv_image_node', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, cam_rgb_callback)

    # Keep the program alive
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()