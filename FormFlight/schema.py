
""" @file schema.py
"""

class ObjectSchema:

    """
    Schematic that classifies a set of objects
    """

    def __init__(self, nobjects, name):
        self.nobjects = nobjects
        self.name = name # 'agents'. 'targets', 'trees' etc.
        self.blueprint = {}

    def set_schema(self, descriptor_type, descriptor_list):

        """
        Populates the blueprint describing each object
        Input:
        - descriptor_type:          string describing the objects within the schema
        - descriptor_list:          list of strings which apply descriptors to each individual
                                    object
        Output:
        """

        assert len(descriptor_list) == self.nobjects

        # assign ID integers to each object
        object_ids = [ID for ID in range(self.nobjects)]
        self.blueprint = self.blueprint.fromkeys(object_ids)

        # apply descriptors to objects
        for ID, descriptor in zip(object_ids, descriptor_list):
            self.blueprint[ID] = {descriptor_type: descriptor}

class DynamicObjectSchema(ObjectSchema):

    """
    Schematic classifying a set of objects which are dynamic
    """

    def __init__(self, nobjects, name):
        super(DynamicObjectSchema, self).__init__(nobjects, name)
        self.descriptor_types = ['dyn_model', 'dyn_type', 'control_pol']

    def set_schema(self, dyn_model_list, dyn_type_list, control_pol_list):

        """
        Generates a schematic for the agents and characteristics of each in a scenario

        - dyn_model_list:   list (of size nobjects) strings representing dynamics models
        - dyn_type_list:    list (of size nobjects) strings representing dynamics model type
        - control_pol_list: list (of size nobjects) strings representing control policies
        """

        assert len(dyn_model_list) == self.nobjects
        assert len(dyn_type_list) == self.nobjects
        assert len(control_pol_list) == self.nobjects

        object_ids = [ID for ID in range(self.nobjects)]
        self.blueprint = self.blueprint.fromkeys(object_ids)

        for ID, dyn_model, dyn_type, control_pol in zip(object_ids, dyn_model_list, dyn_type_list,
                control_pol_list):
            self.blueprint[ID] = {self.descriptor_types[0]: dyn_model, self.descriptor_types[1]: dyn_type,
                    self.descriptor_types[2]: control_pol}

class StaticObjectSchema(ObjectSchema):

    """
    Schematic classifying a set of objects which are static
    """

    def __init__(self, nobjects, name):
        super(StaticObjectSchema, self).__init__(nobjects, name)
        self.descriptor_types = ['dyn_model']

    def set_schema(self, dyn_model_list):

        """
        Generates a schematic for the objects and characteristics of each in a scenario

        - dyn_model_list:   list (of size nagents) strings representing static models
        """

        assert len(dyn_model_list) == self.nobjects

        object_ids = [ID for ID in range(self.nobjects)]
        self.blueprint = self.blueprint.fromkeys(object_ids)

        for ID, dyn_model in zip(object_ids, dyn_model_list):
            self.blueprint[ID] = {self.descriptor_types[0]: dyn_model}

# NOTE NEW
class MultiObjectSchema(ObjectSchema):

    def __init__(self, nobjects, name):
        super(MultiObjectSchema, self).__init__(nobjects, name)

    def set_schema(self, formation):
        self.blueprint['formation'] = formation


