#! /usr/bin/python

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from simple_pid import PID

pid_ang = PID(0.010, 0.002, 0.003, setpoint=256)
pid_spd = PID(0.015, 0.002, 0.01, setpoint=140)

bridge = CvBridge()


def image_callback(msg):
    global val_sim, cv2_img
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)

    pos = Twist()
    pub = rospy.Publisher('/vrep/cmd_vel', Twist, queue_size=10)

    lower_yellow = np.array([20, 150, 150])
    upper_yellow = np.array([30, 255, 255])

    hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours_yellow = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(contours_yellow) > 0:
        res_yellow = cv2.bitwise_and(cv2_img, cv2_img, mask=mask_yellow)
        median_yellow = cv2.medianBlur(res_yellow, 15)

        for i in range(len(contours_yellow)):
            c = contours_yellow[i]
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M["m00"] == 0:
                continue
            else:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(median_yellow, center, 5, (0, 0, 255), -1)
                out_ang = pid_ang(x)
                out_spd = pid_spd(radius)
                print(str(out_ang) + " " + str(out_spd))
                speed = out_spd
                ang = -out_ang
                pos.linear.x = speed
                pos.angular.z = ang
                pub.publish(pos)
    cv2.imshow('mask',mask)
    cv2.imshow('Result', median_yellow)
    cv2.waitKey(1)

def main():
    rospy.init_node('visual_servoing')
    image_topic = "/vrep/image"
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()
