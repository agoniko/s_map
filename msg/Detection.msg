std_msgs/Header header

# list of normalized xyxy points for each detection, need to be reshaped to (n, 4) to get the bounding boxes where n is len(labels)
int32[] boxes
int16[] ids
float32[] scores

# need to be reshaped to (n, h, w)
sensor_msgs/Image masks
sensor_msgs/Image depth

string[] labels

int16 width = 848
int16 height = 480

