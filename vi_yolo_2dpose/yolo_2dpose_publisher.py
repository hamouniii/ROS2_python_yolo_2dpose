#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile,QoSDurabilityPolicy, QoSReliabilityPolicy
from ultralytics import YOLO
from itertools import count
from vi_interfaces.msg import Body2DKeypoints, MultiBody2DKeypoints, BoxAndImage, Box
import torch
import numpy as np


# skeleton order convention:
# 0:  Nose
# 1:  L Eye  
# 2:  R Eye
# 3:  L Ear
# 4:  R Ear 
# 5:  L Shoulder
# 6:  R Shoulder
# 7:  L Elbow 
# 8:  R Elbow
# 9:  L Wrist
# 10: R Wrist  
# 11: L Hip
# 12: R Hip
# 13: L Knee 
# 14: R Knee
# 15: L Ankle 
# 16: R Ankle 

skeleton_lines = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

skeleton_keypoint_name = {
    0:  "Nose",
    1:  "L Eye",  
    2:  "R Eye",
    3:  "L Ear",
    4:  "R Ear",
    5:  "L Shoulder",
    6:  "R Shoulder",
    7:  "L Elbow",
    8:  "R Elbow",
    9:  "L Wrist",
    10: "R Wrist",
    11: "L Hip",
    12: "R Hip",
    13: "L Knee", 
    14: "R Knee",
    15: "L Ankle", 
    16: "R Ankle"
}

 

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('show_visualization', True)
        self.declare_parameter('look_for_missing_keypoints', True)
        self.declare_parameter('confidence_thresh', 0.7)

        self.yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.show_visualization = self.get_parameter('show_visualization').get_parameter_value().bool_value
        self.look_for_missing_keypoints = self.get_parameter('look_for_missing_keypoints').get_parameter_value().bool_value
        self.confidence_thresh = self.get_parameter('confidence_thresh').get_parameter_value().double_value
        

        print("###################################################################")        
        self.log_green(f"type: {type(self.confidence_thresh)}")

        self.get_logger().info(f'YOLO_MODEL_PATH: {self.yolo_model_path}')

        self.yolo_model = YOLO(self.yolo_model_path)  # build a new model from YAML

        qos_profile = QoSProfile(
            depth=10,  # The depth of the history buffer
            durability = QoSDurabilityPolicy.VOLATILE,  # VOLATILE or TRANSIENT_LOCAL
            reliability = QoSReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT or RELIABLE
        )


        self.subscription = self.create_subscription(
            Image,
            '/image',
            self.image_callback_and_send_boxes,
            qos_profile
        )

        # self.subscription = self.create_subscription(
        #     Image,
        #     '/image',  # Replace with the actual image topic name
        #     self.image_callback,
        #     qos_profile)
        
        # self.subscription = self.create_subscription(
        #     Image,
        #     '/image',  # Replace with the actual image topic name
        #     self.fake_image_callback,
        #     qos_profile)


        self.multi_skeleton_publisher_ = self.create_publisher(MultiBody2DKeypoints, 'multi_skeleton_2d', 1)
        self.boxes_and_image_publisher = self.create_publisher(BoxAndImage, 'boxes_and_image', 1)

        self.cv_bridge = CvBridge()
        self.frame_counter = count(0)
        self.skel_id_counter = count(0)
        
        self.boxes = None
        self.masks = None
        self.keypoints = None
        self.probs = None

        self.keypoints_importance_score = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.registered_skel_id_lst = []


        #tmp
        self.fix_on_this_image_idx = 100 
        self.fix_msg = None 




    def generate_skel_id(self):
        return next(self.frame_counter)


    def get_frame_num(self):
        return next(self.frame_counter)


    def log_red(self, message):
        self.get_logger().error(f"\033[91m{message}\033[0m")    
        rclpy.logging.clear_config()


    def log_green(self, message):
        self.get_logger().info(f"\033[92m{message}\033[0m")
        rclpy.logging.clear_config()


    def log_yellow(self, message):
        self.get_logger().debug(f"\033[93m{message}\033[0m")
        rclpy.logging.clear_config()


    def find_missing_keypoints(self, keypoints_2d):
        visible_keypoints_all = []
        for skel_i, one_person_keypoints_2d in enumerate(keypoints_2d): 
            single_skel_visibility_lst = []
            for i in range(len(one_person_keypoints_2d)):
                if(int(one_person_keypoints_2d[i][0].item()) == 0 and int(one_person_keypoints_2d[i][1].item()) == 0):
                   single_skel_visibility_lst.append(0)
                else:
                   single_skel_visibility_lst.append(1)
            visible_keypoints_all.append(single_skel_visibility_lst)
        return visible_keypoints_all


    def find_visible_keypoints(self, skel_idx):
        visibility_lst = []
        for keypoint_idx, keypoint in enumerate(self.keypoints.xy[skel_idx]): 
            if(int(keypoint[0].item()) == 0 and int(keypoint[1].item()) == 0):
                visibility_lst.append(0)
            else:
                visibility_lst.append(1)
        return visibility_lst 


    def visulize_2dskeletons(self, keypoints_2d, boxes, image, txt_color = (0,0,255), line_color = (0, 255, 0), font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.5, font_thickness = 1, line_thickness = 1):
        # self.log_green("Visualizing!")
        for skel_idx in range(len(keypoints_2d.xy)):
            for keypoint_index in range(len(keypoints_2d.xy[0])):
                cv2.putText(image, str(keypoint_index), (int(keypoints_2d.xy[skel_idx][keypoint_index][0].item()), int(keypoints_2d.xy[skel_idx][keypoint_index][1].item())), font, font_scale, txt_color, font_thickness)

            for line in skeleton_lines:
                line_start_point = (int(keypoints_2d.xy[skel_idx][line[0]][0].item()), int(keypoints_2d.xy[skel_idx][line[0]][1].item()))
                line_end_point = (int(keypoints_2d.xy[skel_idx][line[1]][0].item()), int(keypoints_2d.xy[skel_idx][line[1]][1].item()))
                if(line_start_point == (0, 0) or line_end_point == (0,0)):
                    self.log_red(f"Not now showing line between {skeleton_keypoint_name[line[0]]} and {skeleton_keypoint_name[line[1]]}. Reason: missing keypoint.")
                    continue
                cv2.line(image, line_start_point, line_end_point, line_color, line_thickness) 
                     
        self.log_green("Printing boxes:")
        for box in self.boxes:
            cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), line_color, line_thickness)


        cv2.imshow("2D skeletons", image)   
        cv2.waitKey(1)


    def create_keypoits2d_rosmsg(self, skel_idx, skel_id, visibility_lst):
        body_2dkeypoints_msg = Body2DKeypoints()    
        body_2dkeypoints_msg.skeleton_id = skel_id
        
        body_2dkeypoints_msg.x_head = self.keypoints.xy[skel_idx][0][0].item()#nose
        body_2dkeypoints_msg.y_head = self.keypoints.xy[skel_idx][0][1].item()
        body_2dkeypoints_msg.visible_head = visibility_lst[0]
        
        body_2dkeypoints_msg.x_leye = self.keypoints.xy[skel_idx][1][0].item()
        body_2dkeypoints_msg.y_leye = self.keypoints.xy[skel_idx][1][1].item()
        body_2dkeypoints_msg.visible_leye = visibility_lst[1]

        body_2dkeypoints_msg.x_reye = self.keypoints.xy[skel_idx][2][0].item()
        body_2dkeypoints_msg.y_reye = self.keypoints.xy[skel_idx][2][1].item()
        body_2dkeypoints_msg.visible_reye = visibility_lst[2]
        
        body_2dkeypoints_msg.x_lear = self.keypoints.xy[skel_idx][3][0].item()
        body_2dkeypoints_msg.y_lear = self.keypoints.xy[skel_idx][3][1].item()
        body_2dkeypoints_msg.visible_lear = visibility_lst[3]
        
        body_2dkeypoints_msg.x_rear = self.keypoints.xy[skel_idx][4][0].item()
        body_2dkeypoints_msg.y_rear = self.keypoints.xy[skel_idx][4][1].item()
        body_2dkeypoints_msg.visible_rear = visibility_lst[4]
        
        body_2dkeypoints_msg.x_lshoulder = self.keypoints.xy[skel_idx][5][0].item()
        body_2dkeypoints_msg.y_lshoulder = self.keypoints.xy[skel_idx][5][1].item()
        body_2dkeypoints_msg.visible_lshoulder = visibility_lst[5]
        
        body_2dkeypoints_msg.x_rshoulder = self.keypoints.xy[skel_idx][6][0].item()
        body_2dkeypoints_msg.y_rshoulder = self.keypoints.xy[skel_idx][6][1].item()
        body_2dkeypoints_msg.visible_rshoulder = visibility_lst[6]
        
        body_2dkeypoints_msg.x_lelbow = self.keypoints.xy[skel_idx][7][0].item()
        body_2dkeypoints_msg.y_lelbow = self.keypoints.xy[skel_idx][7][1].item()
        body_2dkeypoints_msg.visible_lelbow = visibility_lst[7]
        
        body_2dkeypoints_msg.x_relbow = self.keypoints.xy[skel_idx][8][0].item()
        body_2dkeypoints_msg.y_relbow = self.keypoints.xy[skel_idx][8][1].item()
        body_2dkeypoints_msg.visible_relbow = visibility_lst[8]
        
        body_2dkeypoints_msg.x_lhand = self.keypoints.xy[skel_idx][9][0].item()
        body_2dkeypoints_msg.y_lhand = self.keypoints.xy[skel_idx][9][1].item()
        body_2dkeypoints_msg.visible_lhand = visibility_lst[9]
        
        body_2dkeypoints_msg.x_rhand = self.keypoints.xy[skel_idx][10][0].item()
        body_2dkeypoints_msg.y_rhand = self.keypoints.xy[skel_idx][10][1].item()
        body_2dkeypoints_msg.visible_rhand = visibility_lst[10]
        
        body_2dkeypoints_msg.x_lhip = self.keypoints.xy[skel_idx][11][0].item()
        body_2dkeypoints_msg.y_lhip = self.keypoints.xy[skel_idx][11][1].item()
        body_2dkeypoints_msg.visible_lhip = visibility_lst[11]
        
        body_2dkeypoints_msg.x_rhip = self.keypoints.xy[skel_idx][12][0].item()
        body_2dkeypoints_msg.y_rhip = self.keypoints.xy[skel_idx][12][1].item()
        body_2dkeypoints_msg.visible_rhip = visibility_lst[12]
        
        body_2dkeypoints_msg.x_hip = (body_2dkeypoints_msg.x_lhip + body_2dkeypoints_msg.x_rhip)/2
        body_2dkeypoints_msg.y_hip = (body_2dkeypoints_msg.y_lhip + body_2dkeypoints_msg.y_rhip)/2
        body_2dkeypoints_msg.visible_hip = 1


        body_2dkeypoints_msg.x_lknee = self.keypoints.xy[skel_idx][13][0].item()
        body_2dkeypoints_msg.y_lknee = self.keypoints.xy[skel_idx][13][1].item()
        body_2dkeypoints_msg.visible_lknee = visibility_lst[13]
        
        body_2dkeypoints_msg.x_rknee = self.keypoints.xy[skel_idx][14][0].item()
        body_2dkeypoints_msg.y_rknee = self.keypoints.xy[skel_idx][14][1].item()
        body_2dkeypoints_msg.visible_rknee = visibility_lst[14]
        
        body_2dkeypoints_msg.x_lfoot = self.keypoints.xy[skel_idx][15][0].item()
        body_2dkeypoints_msg.y_lfoot = self.keypoints.xy[skel_idx][15][1].item()
        body_2dkeypoints_msg.visible_lfoot = visibility_lst[15]
        
        body_2dkeypoints_msg.x_rfoot = self.keypoints.xy[skel_idx][16][0].item()
        body_2dkeypoints_msg.y_rfoot = self.keypoints.xy[skel_idx][16][1].item()
        body_2dkeypoints_msg.visible_rfoot = visibility_lst[16]
              
        return body_2dkeypoints_msg            


    def is_valid_keypoint(self, skel_idx):
        conf_np_arr = self.keypoints[skel_idx].conf[0].cpu().numpy()
        score = np.dot(self.keypoints_importance_score, conf_np_arr)/len(conf_np_arr)
        if(score > self.confidence_thresh):
            return True
        else:
            return False


    def create_multi_keypoits2d_rosmsg(self, time_stamp, frame_number, image_height, image_width):
        multi_keypoints2d_msg = MultiBody2DKeypoints()
        multi_keypoints2d_msg.header.stamp = time_stamp
        multi_keypoints2d_msg.frame_number = frame_number
        multi_keypoints2d_msg.img_height = image_height
        multi_keypoints2d_msg.img_width  = image_width

        for idx, single_person_keypoints in enumerate(self.keypoints.xy):
            #single body msg, multiple of the shape MultiBody2DKeypoints msg
            if(not self.is_valid_keypoint(idx)):
                self.log_red("ERROR: Bad skeleton")
                continue

            visibility_lst = self.find_visible_keypoints(idx)

            # codes come here
            body_2dkeypoints_msg = self.create_keypoits2d_rosmsg(skel_idx=idx, skel_id=idx, visibility_lst=visibility_lst)            
            multi_keypoints2d_msg.bodies.append(body_2dkeypoints_msg)

        multi_keypoints2d_msg.total_skeletons = len(multi_keypoints2d_msg.bodies)
        return multi_keypoints2d_msg


    def image_callback_and_send_boxes(self, msg):
        outMsg = BoxAndImage()
        outMsg.header.stamp = msg.header.stamp
        outMsg.bimage = msg
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
        
        results = self.yolo_model(cv_image, device="cuda")  # return a list of Results objects

        for result in results:
            if(len(results) > 1):
                self.log_red("Network Results Size is bigger than 1!")
            self.boxes = result.boxes  # Boxes object for bbox outputs
            self.masks = result.masks  # Masks object for segmentation masks outputs
            self.keypoints = result.keypoints  # Keypoints object for pose outputs
            self.probs = result.probs  # Probs object for classification outputs

        for i, box in enumerate(self.boxes):
            box_msg = Box()
            box_msg.skeleton_id = i
            box_msg.start_x = int(box.xyxy[i][0])
            box_msg.start_y = int(box.xyxy[i][1])
            box_msg.width = int(box.xyxy[i][2] - box.xyxy[i][0])
            box_msg.height = int(box.xyxy[i][3] - box.xyxy[i][1])
            outMsg.boxes.append(box_msg)
        
        outMsg.total_skeletons = len(self.boxes)
        # self.log_green(f"boxes lenght: {len(self.boxes)}")
        self.boxes_and_image_publisher.publish(outMsg)

        if(self.show_visualization):
            self.visulize_2dskeletons(keypoints_2d= self.keypoints, boxes= self.boxes, image= cv_image)
            # cv2.imwrite("/home/heidarshenas/kiu/" + str(len(self.keypoints.xy)) + "kiu_" + str(frame_num) + ".jpg", cv_image)        















    def image_callback(self, msg):

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
        #tmp
        # cv_image = cv2.hconcat([cv_image, cv_image])

        results = self.yolo_model(cv_image, device="cuda")  # return a list of Results objects
        
        for result in results:
            if(len(results) > 1):
                self.log_red("Network Results Size is bigger than 1!")
            self.boxes = result.boxes  # Boxes object for bbox outputs
            self.masks = result.masks  # Masks object for segmentation masks outputs
            self.keypoints = result.keypoints  # Keypoints object for pose outputs
            self.probs = result.probs  # Probs object for classification outputs

                
        # self.log_green(f"confs:\n{keypoints.conf}")
        frame_num = next(self.frame_counter)

        # be careful of *2
        multi_keypoints2d_msg = self.create_multi_keypoits2d_rosmsg(time_stamp=msg.header.stamp, frame_number = frame_num, image_height=msg.height, image_width=msg.width)

        self.multi_skeleton_publisher_.publish(multi_keypoints2d_msg)

        if(self.show_visualization):
            self.visulize_2dskeletons(keypoints_2d= self.keypoints, boxes= self.boxes, image= cv_image)
            cv2.imwrite("/home/heidarshenas/kiu/" + str(len(self.keypoints.xy)) + "kiu_" + str(frame_num) + ".jpg", cv_image)        


    def fake_image_callback(self, msg):
        frame_num = next(self.frame_counter)
        self.log_green(f"Img Num: {frame_num}")

        if(self.fix_on_this_image_idx == frame_num):
            self.fix_msg = msg

        if(frame_num > self.fix_on_this_image_idx):
            msg = self.fix_msg


        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
        
        results = self.yolo_model(cv_image, device="cuda")  # return a list of Results objects
        
        for result in results:
            if(len(results) > 1):
                self.log_red("Network Results Size is bigger than 1!")
            self.boxes = result.boxes  # Boxes object for bbox outputs
            self.masks = result.masks  # Masks object for segmentation masks outputs
            self.keypoints = result.keypoints  # Keypoints object for pose outputs
            self.probs = result.probs  # Probs object for classification outputs

                
        # self.log_green(f"confs:\n{keypoints.conf}")

        # be careful of *2
        multi_keypoints2d_msg = self.create_multi_keypoits2d_rosmsg(time_stamp=msg.header.stamp, frame_number = frame_num, image_height=msg.height, image_width=msg.width)

        self.multi_skeleton_publisher_.publish(multi_keypoints2d_msg)

        if(self.show_visualization):
            self.visulize_2dskeletons(keypoints_2d = self.keypoints, boxes = self.boxes, image= cv_image)
            cv2.imwrite("/home/heidarshenas/kiu/" + str(len(self.keypoints.xy)) + "kiu_" + str(frame_num) + ".jpg", cv_image)        



def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()