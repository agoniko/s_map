#! /usr/bin/env python3

import rospy
from world import World, Obj
from s_map.srv import ManageObject, RemoveObjects, QueryObjects, QueryObjectsResponse, GetAllObjects, GetAllObjectsResponse, CleanUp
from s_map.msg import Object, ObjectList
from std_srvs.srv import Trigger, TriggerResponse
import numpy as np
import json
import rospkg
from threading import Lock
from utils import time_it

PREDICTIONS_PATH = rospkg.RosPack().get_path("s_map") + "/predictions.json"

class WorldManager:
    def __init__(self):
        self.world = World()
        self.lock = Lock()
        self.init_services()

    def init_services(self):
        self.manage_object_service = rospy.Service('manage_object', ManageObject, self.handle_manage_object)
        self.remove_object_service = rospy.Service('remove_objects', RemoveObjects, self.handle_remove_object)
        self.query_objects_service = rospy.Service('query_objects', QueryObjects, self.handle_query_objects)
        self.get_all_objects_service = rospy.Service('get_all_objects', GetAllObjects, self.handle_get_all_objects)
        self.clean_up_service = rospy.Service('clean_up', CleanUp, self.handle_clean_up)
        self.export_predictions_service = rospy.Service('export_predictions', Trigger, self.handle_export_predictions)

    @time_it
    def handle_manage_object(self, req):
        points = np.array(req.object.points).reshape(-1, 3)
        obj = Obj(req.object.id, points, req.object.label, req.object.score, req.object.header.stamp)
        self.world.manage_object(obj)
        return True

    #@time_it
    def handle_remove_object(self, req):
        with self.lock:
            self.world.remove_objects(req.object_ids)
            return True
    
    #@time_it
    def handle_query_objects(self, req):
        with self.lock:
            objects = self.world.query_close_objects_service([req.point.x, req.point.y, req.point.z], req.threshold)

        res = QueryObjectsResponse()
        res.objects = objects
        return res
    
    #@time_it
    def handle_get_all_objects(self, req):
        with self.lock:
            objects = self.world.get_objects()
        if isinstance(objects, ObjectList):
            res = GetAllObjectsResponse()
            res.objects.objects = objects.objects
            res.objects.header = objects.header
        else:
            res = GetAllObjectsResponse()
        
        return res
    
    #@time_it
    def handle_clean_up(self, req):
        with self.lock:
            self.world.clean_up()
            return True
    
    def handle_export_predictions(self, req):
        with self.lock:
            objects = self.world.get_objects()
        if isinstance(objects, ObjectList):
            objects = objects.objects

            preds = {}
            preds['objects'] =  []
            for obj in objects:
                pred = {}
                pred['id'] = obj.id
                pred['label'] = obj.label
                pred['score'] = obj.score
                pred['points'] = obj.points
                pred['OBB'] = obj.bbox
                preds['objects'].append(pred)    
            
            with open(PREDICTIONS_PATH, 'w') as f:
                json.dump(preds, f)
            return TriggerResponse(success=True)
        else:
            return TriggerResponse(success=False)


if __name__ == '__main__':
    rospy.init_node('world_manager')
    manager = WorldManager()
    rospy.spin()
