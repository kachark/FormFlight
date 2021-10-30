
""" @file world.py
"""

from FormFlight import agents

class World():

    def __init__(self, scenario_params):
        self.name = None
        self.scenario_params = scenario_params
        self.multi_object_params = None

        self.objects = {}
        self.dynamic_object_IDs = []
        self.static_object_IDs = []

        self.multi_objects = {}
        self.dynamic_multi_object_IDs = []
        self.static_multi_object_IDs = []

        self.world_state = None
        self.world_time = 0

    def set_name(self, name):
        self.name = name

    def generate(self, **kwargs):

        """
        Creates the Objects (static and dynamic) within the World as defined by a scenario
        """

        i = 0

        # create dynamic objects and assign global ID
        for schema in self.scenario_params['dynamic_objects']:
            for blueprint_id, object_info in schema.blueprint.items():
                object_model = object_info['dyn_model']

                object_ID = i
                ag = agents.Agent(object_model)
                ag.info = object_info
                ag.ID = object_ID
                self.dynamic_object_IDs.append(object_ID)
                self.objects[object_ID] = ag
                i += 1

        # create static objects and assign global ID
        for schema in self.scenario_params['static_objects']:
            for blueprint_id, object_info in schema.blueprint.items():
                object_model = object_info['dyn_model']

                object_ID = i
                ag = agents.Point(object_model)
                ag.info = object_info
                ag.ID = object_ID
                self.static_object_IDs.append(object_ID)
                self.objects[object_ID] = ag
                i += 1

        # check if need to group up objects
        multi_object_schemas_list = kwargs['multi_object_schemas_list']
        if multi_object_schemas_list:
            objects_available = len(self.objects)
            index = 0
            self.multi_object_params = multi_object_schemas_list
            I = 0
            for multi_object_schema in self.multi_object_params['dynamic_objects']:
                objects_needed = multi_object_schema.nobjects
                initial_formation = multi_object_schema.blueprint['formation']
                name = multi_object_schema.name

                object_list = [None]*objects_needed
                for n in range(objects_needed):
                    object_list[n] = self.objects[index]
                    index += 1

                group_ID = I
                mas = agents.MultiAgentSystem(name, object_list)
                mas.ID = group_ID
                mas.formation = initial_formation

                self.dynamic_multi_object_IDs.append(group_ID)
                self.multi_objects[group_ID] = mas
                I += 1

            for multi_object_schema in self.multi_object_params['static_objects']:
                objects_needed = multi_object_schema.nobjects
                initial_formation = multi_object_schema.blueprint['formation']
                name = multi_object_schema.name

                object_list = [None]*objects_needed
                for n in range(objects_needed):
                    object_list[n] = self.objects[index]
                    index += 1

                group_ID = I
                mas = agents.MultiAgentSystem(name, object_list)
                mas.ID = group_ID
                mas.formation = initial_formation

                self.static_multi_object_IDs.append(group_ID)
                self.multi_objects[group_ID] = mas
                I += 1

    def get_object_world_state_index(self, ID):

        """
        Returns the start and end indices of the world state for an Object
        Input:
        - ID:           integer corresponding to the global ID on a world object
        Output:
        """

        start_index = None
        end_index = None

        object_i = self.objects[ID]
        dx_i = object_i.dx
        if ID == 0:
            start_index = 0
            end_index = dx_i
        elif ID > 0:
            start_index = (ID * dx_i)
            end_index = start_index + dx_i

        return start_index, end_index

    def get_multi_object(self, name):

        for group_ID, multi_object in self.multi_objects.items():
            if name == multi_object.name:
                return multi_object

        return None

    def update_world_state(self, new_state):
        # NOTE want to avoid copying
        pass

    def update_world_time(self, new_time):
        pass

    def create_ID(self):
        pass

