#! /usr/bin/env python3

import rospy
from world import World, Obj
from s_map.srv import ManageObject, RemoveObjects, QueryObjects, QueryObjectsResponse, GetAllObjects, GetAllObjectsResponse
from s_map.msg import Object, ObjectList
import numpy as np

class WorldManager:
    def __init__(self):
        self.world = World()
        self.init_services()

    def init_services(self):
        self.manage_object_service = rospy.Service('manage_object', ManageObject, self.handle_manage_object)
        self.remove_object_service = rospy.Service('remove_objects', RemoveObjects, self.handle_remove_object)
        self.query_objects_service = rospy.Service('query_objects', QueryObjects, self.handle_query_objects)
        self.get_all_objects_service = rospy.Service('get_all_objects', GetAllObjects, self.handle_get_all_objects)

    def handle_manage_object(self, req):
        points = np.array(req.object.points).reshape(-1, 3)
        obj = Obj(req.object.id, points, req.object.label, req.object.score, req.object.header.stamp)
        self.world.manage_object(obj)
        return True

    def handle_remove_object(self, req):
        self.world.remove_objects(req.object_ids)
        return True

    def handle_query_objects(self, req):
        objects = self.world.query_by_distance([req.point.x, req.point.y, req.point.z], req.threshold)
        res = QueryObjectsResponse()
        res.objects = [self.convert_obj_to_msg(obj) for obj in objects]
        return res
    
    def handle_get_all_objects(self, req):
        objects = self.world.get_objects()
        if isinstance(objects, ObjectList):
            res = GetAllObjectsResponse()
            res.objects.objects = objects.objects
            res.objects.header = objects.header
        else:
            res = GetAllObjectsResponse()
        
        return res

    def convert_obj_to_msg(self, obj):
        msg = Object()
        msg.id = obj.id
        msg.points = obj.points
        msg.label = obj.label
        msg.score = obj.score
        msg.timestamp = obj.timestamp
        return msg

if __name__ == '__main__':
    rospy.init_node('world_manager')
    manager = WorldManager()
    rospy.spin()
