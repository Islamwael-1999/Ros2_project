#!/usr/bin/env python3
import rclpy
from rclpy.node import  Node
from example_interfaces.msg import String
class motion_plan(Node):
    
    def callback(self,msg):
        #pos=self.get_logger().info(msg.data)
        pos=int(msg.data)
        directions=String()
        if pos == 0:
            directions.data="no gate detected"
        else:
            if (pos-1)//3==0:
                str1="UP"
            elif (pos-1)//3==1:
                str1=""
            elif (pos-1)//3==2:
                str1="down"
            if (pos-1)%3==0:
                str2=" left"
            elif (pos-1)%3==1:
                str2=" forward"
            elif (pos-1)%3==2:
                str2=" right"
            directions.data=str1+str2
        self.publisher_.publish(directions)
       

    
    def __init__(self):
       super().__init__("motion_plan")
       self.subscriber_=self.create_subscription(String,"GATE_POSITION",self.callback,10)
       self.publisher_=self.create_publisher(String,"DIRECTION",10)
def main(args=None):
    rclpy.init(args=args)  
    node=motion_plan()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__=="__main__":
    main()