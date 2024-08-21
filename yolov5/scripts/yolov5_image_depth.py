#!/usr/bin/env python3
# 实时显示摄像头的深度图像
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def cam_depth_callback(msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV image
        depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Apply colormap to depth image for better visualization
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    # Display the depth image
    cv2.imshow("Depth Image", depth_colormap)
    cv2.waitKey(1)

def main():
    rospy.init_node('cv_image_node', anonymous=True)
    rospy.Subscriber("/camera/depth/image_raw", Image, cam_depth_callback)

    # Keep the program alive
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()