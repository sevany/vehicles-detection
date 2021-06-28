import numpy as np
import os
import six.moves.urllib as urllib
import sys
import cv2
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display


from jsonya import id2name
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#from keras.preprocessing.image import save_img


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

##############################################################################

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        print('shit!!')

physical_devices = tf.config.experimental.list_physical_devices('GPU')

##########################################################################LOADER###########################################################################

#   return model
##########################################################################################
title = "Twistcode Object detection"

MODEL_NAME = 'model/faster_rcnn_inception_v2_coco_2018_01_28'
IMAGE_NAME = 'images/image8.jpg'
############################################################################################Loading label map#############################################333
# Grab path to current working directory
CWD_PATH = os.getcwd()
THRESHOLD = 0.5
NUM_CLASSES = 50
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
 
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images/')
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
# TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

# model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
# detection_model = load_model(model_name)

# print(detection_model.inputs)
# print(detection_model.output_dtypes)
# print(detection_model.output_shapes)
#####################################################################################################

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value

#########################################################################################
image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})


vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8
    )
    # min_score_thresh=0.60)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# return image
####################################################################

height, width, _ = image_rgb.shape
results = []
for idx, class_id in enumerate(classes[0]):
    conf = scores[0, idx]
    if conf > THRESHOLD:
        bbox = boxes[0, idx]
        ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
        results.append({"name": id2name[class_id],
                        "conf": str(conf),
                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
        })

print({"results":results})
# return {"results":results}
##################################################################

filename = 'test1.jpg'
# All the results have been drawn on image. Now display the image.
cv2.imshow(IMAGE_NAME, image)
# cv2.imshow('image4.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imwrite(filename, image)

# Press any key to close the image
cv2.waitKey(0)

# # Clean up
cv2.destroyAllWindows()
