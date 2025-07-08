**object detection using YOLOv5 + PyTorch + ROS 2 + OpenCV.**

**Dependencies You Need to Install (once):**
pip install torch torchvision torchaudio
pip install opencv-python
pip install pandas
pip install ultralytics
sudo apt install ros-humble-cv-bridge  # Or your ROS 2 version

python code::

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import json

class YoloObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = 0.5  # Minimum confidence threshold

        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for detection results
        self.publisher = self.create_publisher(String, '/shelf_objects', 10)

        self.get_logger().info('YOLO Object Detection Node Started.')

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLO detection
        results = self.model(cv_image)

        # Extract detections
        detections = results.pandas().xyxy[0]
        objects = []

        for _, row in detections.iterrows():
            obj = {
                "label": row['name'],
                "confidence": float(row['confidence']),
                "bbox": {
                    "xmin": int(row['xmin']),
                    "ymin": int(row['ymin']),
                    "xmax": int(row['xmax']),
                    "ymax": int(row['ymax'])
                }
            }
            objects.append(obj)

            # Optional: draw bounding box for debug
            cv2.rectangle(cv_image, (obj["bbox"]["xmin"], obj["bbox"]["ymin"]),
                          (obj["bbox"]["xmax"], obj["bbox"]["ymax"]), (0, 255, 0), 2)
            cv2.putText(cv_image, obj["label"], (obj["bbox"]["xmin"], obj["bbox"]["ymin"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish detected objects as JSON
        msg_out = String()
        msg_out.data = json.dumps({"objects": objects})
        self.publisher.publish(msg_out)

        # Optional: show debug image
        cv2.imshow("Detected Objects", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

