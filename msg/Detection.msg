std_msgs/Header header

# list of normalized xyxy points for each detection, need to be reshaped to (n, 4) to get the bounding boxes where n is len(labels)
float32[] boxes
int16[] ids
float32[] scores

# need to be reshaped to (n, h, w)
bool[] masks

string[] labels

int16 width = 848
int16 height = 480

