# Basic ROS 2 program to publish real-time streaming 
# video from your built-in webcam
  
# ROS2-based libraries
import cv2 # OpenCV library
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from std_msgs.msg import String

# ML-based libraries
import numpy as np
from tensorflow import keras
 
class ImagePublisher(Node):
  """
  Create an ImagePublisher class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_publisher')
    
    # Loading model
    self.model = keras.models.load_model("saved_model")
      
    # Create the publisher. This publisher will publish an Image
    # to the video_frames topic. The queue size is 10 messages.
    # 'video_frames' is the topic name that this node is publishing data to.
    # self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
    self.publisher_ = self.create_publisher(String, 'video_frames', 10)
      
    # We will publish a message every 0.1 seconds
    timer_period = 0.1  # seconds
      
    # Create the timer
    self.timer = self.create_timer(timer_period, self.timer_callback)
         
    # Create a VideoCapture object
    # The argument '0' gets the default webcam.
    self.cap = cv2.VideoCapture(0)
         
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
   
  def timer_callback(self):
    """
    Callback function.
    This function gets called every 0.1 seconds.
    """
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame = self.cap.read()
    
    # The original frame shape is: (480, 640, 3)
    # The CNN model expects: (64, 64, 3)
    # opencv already gives us the frame as a numpy array.
    frame = cv2.resize(frame, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    frame = np.expand_dims(frame, 0)
    
    # Debugging purposes if needed.
    # print("New frame shape:")
    # print(frame.shape)
    
    classes=["dirt", "sand", "wet"]
    
    prediction = self.model.predict(frame)
    class_ix = np.argmax(prediction, axis=1)
    prediction = classes[class_ix[0]]
          
    msg = String()
    msg.data = prediction
    self.publisher_.publish(msg)
 
    # Display the message on the console
    self.get_logger().info('Publishing prediction')
  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_publisher = ImagePublisher()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_publisher)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_publisher.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
