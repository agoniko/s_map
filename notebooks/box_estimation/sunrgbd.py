import os, sys
import json
import numpy as np
import glob

#-------------------------------|Helper Functions|------------------------------

def load_json(pattern):
    try:
        with open(glob.glob(pattern)[0]) as f:
            return json.load(f)
    except IndexError:
        return None
    except json.decoder.JSONDecodeError:
        return None

def load_numpy(file_path):
    try:
        return np.loadtxt(file_path)
    except OSError:
        return None

def load_text(file_path):
    try:
        with open(file_path) as f:
            return f.read().strip()
    except OSError:
        return None

def get_first_file_path(pattern):
    files = glob.glob(pattern)
    if files:
        return files[0]
    else:
        return None

def write_obj(boundshapes, filename):
    with open(filename, 'w') as file:
        vertex_offset = 1
        for boundshape in boundshapes:
            for point in boundshape:
                file.write(f"v {point[0]} {point[1]} {point[2]}\n")
            for i in range(0, len(boundshape) - 2, 2):
                file.write(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 3 + vertex_offset} {i + 2 + vertex_offset}\n")
            if len(boundshape) > 4:
                file.write(f"f {len(boundshape) - 2 + vertex_offset} {len(boundshape) - 1 + vertex_offset} {1 + vertex_offset} {vertex_offset}\n")
            ymin_face_indices = ' '.join(str(i + vertex_offset) for i in range(0, len(boundshape), 2))
            file.write(f"f {ymin_face_indices}\n")
            ymax_face_indices = ' '.join(str(i + vertex_offset) for i in range(1, len(boundshape), 2))
            file.write(f"f {ymax_face_indices}\n")
            vertex_offset += len(boundshape)

#-------------------------------------------------------------------------------

class SunRGBD_Record:
    def __init__(self, record_dir):
        super(SunRGBD_Record, self).__init__()
        self.record_dir = record_dir
        self.annot_3d = load_json(os.path.join(self.record_dir, "annotation3Dfinal", "*.json"))
        self.annot_2d = load_json(os.path.join(self.record_dir, "annotation2Dfinal", "*.json"))
        self.annot_room = load_json(os.path.join(self.record_dir, "annotation3Dlayout", "*.json"))
        self.intrinsics = load_numpy(os.path.join(self.record_dir, "intrinsics.txt"))
        extrinsics_files = glob.glob(os.path.join(self.record_dir, "extrinsics", "*.txt"))
        if extrinsics_files:
            self.extrinsics = load_numpy(extrinsics_files[0])
        else:
            self.extrinsics = None
        self.scene_descrip = load_text(os.path.join(self.record_dir, "scene.txt"))
        self.image_path = get_first_file_path(os.path.join(self.record_dir, "image", "*.*"))
        self.depth_path = get_first_file_path(os.path.join(self.record_dir, "depth", "*.*"))

    def get_number_of_objects(self):
        return len(self.annot_3d.get('objects', []))

    def get_object_type_by_index(self, index):
        if self.annot_3d['objects'][index] is None:
            return None
        return self.annot_3d['objects'][index]['name']

    def get_object_polygon_by_index(self, index):
        object_info = self.annot_3d['objects'][index]
        polygon = object_info['polygon'][0]
        points = []
        for x, z in zip(polygon['X'], polygon['Z']):
            points.append((x, polygon['Ymin'], z))
            points.append((x, polygon['Ymax'], z))
        return points

    def is_polygon_a_box(self, index):
        object_info = self.annot_3d['objects'][index]
        polygons = object_info['polygon']
        return polygons[0]['rectangle']

    def get_room_polygon(self):
        room_info = self.annot_room['objects'][0]
        polygon = room_info['polygon'][0]
        points = []
        for x, z in zip(polygon['X'], polygon['Z']):
            points.append((x, polygon['Ymin'], z))
            points.append((x, polygon['Ymax'], z))
        return points

    def get_segments_2d(self):
        num_segments = len(self.annot_2d["frames"][0]["polygon"])
        segments = []
        labels = []
        for i in range(num_segments):
            x = self.annot_2d["frames"][0]["polygon"][i]["x"]
            y = self.annot_2d["frames"][0]["polygon"][i]["y"]
            obj_pointer = self.annot_2d["frames"][0]["polygon"][i]["object"]
            points = np.transpose(np.array([x,y], np.int32))
            segments.append(points)
            labels.append(self.annot_2d['objects'][obj_pointer]["name"])
        return (labels, segments)

    def __repr__(self):
        return (f"SunRGBD_Record(\n"
                f"  record_dir='{self.record_dir}',\n"
                f"  annot_3d_keys={list(self.annot_3d.keys()) if self.annot_3d else 'None'},\n"
                f"  annot_2d_keys={list(self.annot_2d.keys()) if self.annot_2d else 'None'},\n"
                f"  annot_room_keys={list(self.annot_room.keys()) if self.annot_room else 'None'},\n"
                f"  intrinsics_shape={self.intrinsics.shape if self.intrinsics is not None else 'None'},\n"
                f"  extrinsics_shape={self.extrinsics.shape if self.extrinsics is not None else 'None'},\n"
                f"  scene_descrip='{self.scene_descrip if self.scene_descrip is not None else 'None'}',\n"
                f"  image_path='{self.image_path if self.image_path is not None else 'None'}',\n"
                f"  depth_path='{self.depth_path if self.depth_path is not None else 'None'}'\n"
                f")")



class SunRGBD_Dataset:
    def __init__(self, ds_dir):
        super(SunRGBD_Dataset, self).__init__()
        self.root_dir = ds_dir
        self.records = self._setup_records()

    def _setup_records(self):
        all_entries = os.listdir(self.root_dir)
        records = []
        for entry in all_entries:
            fpath = os.path.join(self.root_dir, entry)
            if os.path.isdir(fpath):
                records.append(SunRGBD_Record(fpath))
        return records

    def __len__(self):
        return len(self.records)


#----------------------------------|Testing|------------------------------------
import cv2
import random

def show_record_images(rec):
    img_rgb = cv2.imread(rec.image_path)
    img_d = cv2.imread(rec.depth_path)
    img_annot = np.array(img_rgb, copy=True)
    labels, segments = rec.get_segments_2d()
    for i in range(0, len(segments)):
        color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        cv2.fillPoly(img_annot, [segments[i]], color)
    for i in range(0, len(segments)):
        data = segments[i]
        centroid = np.mean(data,axis=0)
        cv2.putText(img_annot, labels[i], (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,0,0], 2)
    cv2.imshow("Detph Image", img_d)
    cv2.imshow("RGB Image", img_rgb)
    cv2.imshow("Annotated Image", img_annot)
    cv2.waitKey(0)


DATASET_PATH = "SUNRGBD/kv2/align_kv2"

def main():
    print("Testing dataset code.")
    assert len(DATASET_PATH) > 0, "You forgot to change the file path."
    ds = SunRGBD_Dataset(DATASET_PATH)
    print("Dataset size:  ", len(ds))
    rec = ds.records[0]
    print(rec)
    print()
    n = rec.get_number_of_objects()
    print(rec.record_dir)
    print("Num objs:   ", n)
    print("Room Bounds:")
    polygons = []
    for x, y, z in rec.get_room_polygon():
        print("  ", (x, y, z))
    for oi in range(n):
        print()
        print("Index:  ", oi, '/', n)
        print(len(rec.annot_3d['objects']))
        ot = rec.get_object_type_by_index(oi)
        print("Type:  ", ot)
        if ot is not None:
            polygons.append(rec.get_object_polygon_by_index(oi))
            print("Bounds:")
            for x, y, z in rec.get_object_polygon_by_index(oi):
                print("  ", (x, y, z))
            print("Is a box?:  ", rec.is_polygon_a_box(oi))
    write_obj(polygons, "test_objs.obj")
    write_obj([rec.get_room_polygon()], "test_room.obj")
    show_record_images(rec)


if __name__ == '__main__':
    main()
