
""" @file system.py
"""

from decimal import Decimal
from time import time, process_time
import scipy.integrate as scint
import numpy as np

from FormFlight import assignments
from FormFlight import controls


################################
## Big Systems
################################
class System:

    """ System parent class
    """

    def __init__(self, scenario):

        """ System constructor
        """

        self.scenario = scenario

class OneVOne(System):

    """ System representing scenario consisting of one agent to one target engagements

    """

    def __init__(self, scenario, world_i):

        """ Constructor for OneVOne System
        """

        super(OneVOne, self).__init__(scenario)
        self.world = world_i

        self.costs = []

    def compute_assignments(self, t, world_state, system_of_interest, targettable_set, collisions):

        """ Compute assignments between agent and target swarms

        Does not perfrom assignments for agents that are collided with targets

        """

        nagents = system_of_interest.nagents
        ntargets = targettable_set.nagents

        # collect agents and target states that are not predicted to collide : (state, object)
        living_agents = []
        for agent in system_of_interest.agent_list:
            object_ID = agent.ID
            for c in collisions: # if this agent predicted to collide, skip assignment
                if object_ID == c[0]:
                    continue

            start_ind, end_ind = self.world.get_object_world_state_index(object_ID)
            living_agents.append( (world_state[start_ind:end_ind], agent) )

        living_targets = []
        for target in targettable_set.agent_list:
            object_ID = target.ID
            for c in collisions: # if this target predicted to collide, don't use in assignment
                if object_ID == c[1]:
                    continue

            start_ind, end_ind = self.world.get_object_world_state_index(object_ID)
            living_targets.append( (world_state[start_ind:end_ind], target) )

        # perform assignment
        decision_maker = system_of_interest.decision_maker
        assignments, cost = decision_maker.assignment(t, living_agents, living_targets)

        return assignments, cost

    def pre_process(self, t0, world_state, collisions):

        """ System pre-processor

        Perform functions prior to starting engine loop
        """

        # assign targets to static states naively

        # get target terminal state group
        target_terminal_points_group = self.world.get_multi_object('Region')
        terminal_point_list = target_terminal_points_group.agent_list

        # get 'Targets'
        target_mas = self.world.get_multi_object('Target_MAS')
        target_list = target_mas.agent_list

        # get 'Target terminal points'
        target_terminal_mas = self.world.get_multi_object('Region')
        target_terminal_list = target_terminal_mas.agent_list

        # NOTE world_state has all states together
        # # assign targets to static states
        # for (target, terminal_point) in zip(target_list, terminal_point_list): 
        #     ID_1 = terminal_point.ID
        #     start_ind, end_ind = self.world.get_object_world_state_index(ID_1)
        #     terminal_state = world_state[start_ind: end_ind]

        #     ID_2 = target.ID
        #     target_indices = self.world.get_object_world_state_index(ID_1)
        #     # update target controller

        # NOTE hardcoded static states not ideal
        for (target, terminal_point) in zip(target_list, target_terminal_list):
            if target.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                term_start_ind, term_end_ind = self.world.get_object_world_state_index(terminal_point.ID)
                terminal_state = world_state[term_start_ind:term_end_ind]
                target.pol.set_const(terminal_state)
            else:
                pass

        # # TEST
        # # NOTE SCENARIO SPECIFIC SETUP
        # # agent intent policy
        # agent_mas = self.world.get_multi_object('Agent_MAS')
        # agent_list = agent_mas.agent_list
        # for agent in agent_list:
        #     new_pol = controls.LinearIntentPolicy(agent.pol)
        #     agent.pol = new_pol

    def update(self, t0, dt, tick, world_state0, collisions):

        """ Computes assignments at assignment epoch and advances dynamics per engine tick

        Input:
        - t0:           start time of integration
        - world_state:  agent, target, target terminal states at start time of integration
        - dt:           engine time step size
        - tick:
        - collisions: 

        Output:
        return tout, yout, assign_out, diagnostics
        - tout:         time integrated over between [t0, t0+dt]
        - yout:         agent states, target states between [t0, t0+dt]
        - assign_out:   index assignments between agent_i and target_j
        - diagnostics:  diagnostics recorded between [t0, t0+dt]

        """

        systems_of_interest = ['Agent_MAS', 'Target_MAS']
        multi_objects = self.world.multi_objects

        # NOTE hardcoded for now
        agent_targettable_set_name = 'Target_MAS'
        target_targettable_set_name = 'Region'

        # engagement
        engagement = [('Agent_MAS', 'Target_MAS')]

        # measure assignment execution time
        start_assign_time = process_time()

        # perform decision with multi-agent systems of interest
        for pair in engagement:
            system_name = pair[0]
            targettable_set = pair[1]

            # multi-agent system (MAS)
            mas = self.world.get_multi_object(system_name)
            agent_list = mas.get_agent_list()
            assignment_epoch = mas.decision_epoch
            decision_maker = mas.decision_maker

            # targettable set
            target_mas = self.world.get_multi_object(targettable_set)

            if not decision_maker:
                break

            assignment = None
            if t0 == 0:
                # the MAS independently performs the assignment
                # assignment, cost = mas.compute_decision(t0, target_mas, mas_state, collisions)

                # the System (ie. central authority) performs the decision
                assignment, cost = self.compute_assignments(t0, world_state0, mas,
                        target_mas, collisions)
                mas.current_decision = assignment

            # 'DYN' algorithm needs to be computed only once
            if t0 > 0 and decision_maker.__class__.__name__ != 'AssignmentDyn':
                if tick % assignment_epoch == 0:
                    print(system_name, " ------> ASSIGNMENT AT: ", t0)

                    # the MAS independently performs the assignment
                    # assignment, cost = mas.decide(t0, mas_state, collisions)

                    # the System (ie. central authority) performs the decision
                    assignment, cost = self.compute_assignments(t0, world_state0, mas,
                            target_mas, collisions)
                    mas.current_decision = assignment

                else:
                    assignment = mas.current_decision
            else:
                assignment = mas.current_decision

            # after assignment done:
            # update tracking control policy
            if t0 == 0:
                for object_asst in assignment:
                    agent_ID = object_asst[0]
                    target_ID = object_asst[1]

                    agent = self.world.objects[agent_ID]
                    target = self.world.objects[target_ID]

                    # update the linear augmented tracker
                    if agent.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                        # NOTE assumes that agents have tracking controllers
                        target_Acl = target.pol.get_closed_loop_A()
                        target_gcl = target.pol.get_closed_loop_g()
                        agent.pol.track(t0, target_ID, target_Acl, target_gcl)
                        # TEST inten learning
                        # agent.pol.track(t0, target)
                        # if not np.array_equal(agent.pol.agent_pol.p, agent.pol.agent_pol.p.T):
                        #     import ipdb; ipdb.set_trace()
                        # if agent.pol.agent_pol.p.all() == agent.pol.agent_pol.p.T.all():
                        #     import ipdb; ipdb.set_trace()
                    elif agent.pol.__class__.__name__ == 'NonlinearController':
                        agent.pol.track(t0, target_ID)

            if t0 > 0 and decision_maker.__class__.__name__ != 'AssignmentDyn':
                for object_asst in assignment:
                    agent_ID = object_asst[0]
                    target_ID = object_asst[1]

                    agent = self.world.objects[agent_ID]
                    target = self.world.objects[target_ID]

                    # update the linear augmented tracker
                    if agent.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                        # NOTE assumes that agents have tracking controllers
                        target_Acl = target.pol.get_closed_loop_A()
                        target_gcl = target.pol.get_closed_loop_g()
                        agent.pol.track(t0, target_ID, target_Acl, target_gcl)
                        # TEST inten learning
                        # agent.pol.track(t0, target)
                    elif agent.pol.__class__.__name__ == 'NonlinearController':
                        agent.pol.track(t0, target_ID)


        # measure assignment execution time
        elapsed_assign_time = process_time() - start_assign_time

        # record cost-to-go (# TODO is this true anymore?)
        # self.costs.append(cost)

        # if cost is not None:
        #     print("TIME: ", t0, "ASST TYPE: ", self.apol.__class__.__name__)
            # print("TIME: ", t0, "COST: ", cost, "ASST: ", assignment)

        print("TIME: ", t0, "ASST TYPE: ", decision_maker.__class__.__name__)
        print("ASST: ", [agent.pol.tracking for agent in self.world.get_multi_object('Agent_MAS').agent_list] )

        # propogates dynamic objects
        def dyn(t, x):

            dxdt = np.zeros(x.shape)

            dynamic_object_IDs = self.world.dynamic_object_IDs

            # evolve systems of interest since they interact with eachother (ie. feedforward)
            evaluated = set()
            for groups in engagement:
                system_name = groups[0]
                targettable_set = groups[1]

                mas = self.world.get_multi_object(system_name)
                target_mas = self.world.get_multi_object(targettable_set)

                soi_dynamic_objects = mas.agent_list
                target_dynamic_objects = target_mas.agent_list

                assignment = mas.current_decision
                for pair in assignment:
                    agent_ID = pair[0]
                    target_ID = pair[1]

                    evaluated.add(agent_ID)
                    evaluated.add(target_ID)

                    agent = self.world.objects[agent_ID]
                    target = self.world.objects[target_ID]

                    ag_start_ind, ag_end_ind = self.world.get_object_world_state_index(agent_ID)
                    t_start_ind, t_end_ind = self.world.get_object_world_state_index(target_ID)

                    agent_state = x[ag_start_ind: ag_end_ind]
                    target_state = x[t_start_ind: t_end_ind]

                    u_target = target.pol.evaluate(t, target_state)

                    # evaluate controls
                    u = None
                    if agent.pol.__class__.__name__ == 'LinearFeedbackAugmented':
                        u = agent.pol.evaluate(t, agent_state, target_state, feedforward=u_target)
                    elif agent.pol.__class__.__name__ == 'NonlinearController':
                        u = agent.pol.evaluate(t, agent_state, target_state)

                    if not bool(collisions):
                        dxdt[ag_start_ind:ag_end_ind] = agent.dyn.rhs(t, agent_state, u)
                        dxdt[t_start_ind:t_end_ind] = target.dyn.rhs(t, target_state, u_target)
                    else: # don't propogate dynamics
                        for c in collisions: # collisions = set of tuples
                            if agent_ID == c[0] or target_ID == c[1]:
                                # break # or continue?
                                continue

            leftover = [x for x in dynamic_object_IDs if x not in evaluated]

            # assumes all agents assigned and only targets leftover to evolve
            for target_ID in leftover:
                target = self.world.objects[target_ID]
                t_start_ind, t_end_ind = self.world.get_object_world_state_index(target_ID)
                target_state = x[t_start_ind: t_end_ind]
                u_target = target.pol.evaluate(t, target_state)
                if not bool(collisions):
                    dxdt[t_start_ind:t_end_ind] = target.dyn.rhs(t, target_state, u_target)
                else: # don't propogate dynamics
                    for c in collisions: # collisions = set of tuples
                        if target_ID == c[1]:
                            # break # or continue?
                            continue

            return dxdt


        tspan = (t0, t0+dt)

        # measure dynamics execution time
        start_dynamics_time = process_time()

        bunch = scint.solve_ivp(dyn, tspan, world_state0, method='BDF', rtol=1e-6, atol=1e-8)

        # measure dynamics execution time
        elapsed_dynamics_time = process_time() - start_dynamics_time

        tout = bunch.t
        yout = bunch.y.T
        # assign_out = np.tile(assignment, (tout.shape[0], 1))
        # NOTE assumes that assignment is ordered by object_ID
        assign_out = np.tile([pair[1] for pair in assignment], (tout.shape[0], 1))

        # **** system diagnostics
        assign_comp_cost = np.tile(elapsed_assign_time, (tout.shape[0], 1))
        assign_comp_cost[1:tout.shape[0]] = 0

        dynamics_comp_cost = np.tile(elapsed_dynamics_time, (tout.shape[0], 1))
        dynamics_comp_cost[1:tout.shape[0]] = 0

        diagnostics = [assign_comp_cost, dynamics_comp_cost]
        # **** system diagnostics

        return tout, yout, assign_out, diagnostics

        # print(tout, yout)
        # exit(1)

