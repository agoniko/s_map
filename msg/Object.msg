# Object.msg
Header header  # Standard metadata for higher-level stamped data types
int32 id      # Unique identifier for the object
float32[] points  # Flattened list of points (x, y, z)
string label   # Label of the object
float32 score  # Detection score
float32[] bbox   #Oriented Bounding Box (OBB) of shape (8, 3)