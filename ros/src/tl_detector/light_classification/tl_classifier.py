from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
import os

# NUM_CLASSES      = 4
# MIN_SCORE_THRESH = 0.3

backup_parts = [os.getcwd(), '..', '..', '..', 'classifier', 'ssd_inception_v2_coco_sim', 'frozen_inference_graph.pb'] 
backup_path  = os.path.join( *backup_parts)

class TLClassifier(object):
    def __init__(self):
        rospy.logwarn("backup_path: {}".format(backup_path) )
        
        self.model_path      = rospy.get_param('model_path', backup_path)
        self.detection_graph = tf.Graph()        
       
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile( self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.tf_session = tf.Session(graph= self.detection_graph)

        self.image_tensor      = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes   = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores  = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        #self.num_detections    = self.detection_graph.get_tensor_by_name('num_detections:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)

            # Run inference
            (boxes, scores, classes) = self.tf_session.run( 
                         [self.detection_boxes, self.detection_scores, self.detection_classes] ,
                         feed_dict={ self.image_tensor: image_np_expanded})
        
            boxes   = np.squeeze(boxes)
            scores  = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            if scores is not None and classes is not None:
                if scores[0] > 0.3: # MIN_SCORE_THRESH
                    if classes[0] == 1:
                        return TrafficLight.GREEN
                    elif classes[0] == 2:
                        return TrafficLight.RED
                    elif classes[0] == 3:
                        return TrafficLight.YELLOW

            return TrafficLight.UNKNOWN

