import gym
import copy
import math
import heapq
import logging
import functools
import numpy as np
import time
import pandas as pd
from optical_rl_gym.utils import Service, Path
from .optical_network_env import OpticalNetworkEnv


# The new class adds some features in the metrics for explainability
class RMSAEnv1(OpticalNetworkEnv):

    metadata = {
        'metrics': ['service_blocking_rate', 'episode_service_blocking_rate',
                    'bit_rate_blocking_rate', 'episode_bit_rate_blocking_rate', 
                    'num_C_selected', 'num_L_selected', 'num_S_selected', 'num_E_selected',
                    'num_path1_selected', 'num_path2_selected', 'num_path3_selected', 'num_path4_selected',
                    'num_path5_selected', 'total_path_length_per_episode']
    }

    def __init__(self, topology=None,
                 episode_length=1000,
                 load=10,
                 mean_service_holding_time=10800.0,
                 num_spectrum_resources=[10, 100],
                 node_request_probabilities=None,
                 bit_rate_lower_bound=25,
                 bit_rate_higher_bound=400,
                 seed=None,
                 k_paths=5,
                 filename = '',
                 allow_rejection=False,
                 reset=True):
        super().__init__(topology,
                         episode_length=episode_length,
                         load=load,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         allow_rejection=allow_rejection,
                         k_paths=k_paths)
        assert 'modulations' in self.topology.graph
        # specific attributes for MB optical networks
        self.bands = 4
        self.band = 0
        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0

        self.bit_rate_lower_bound = bit_rate_lower_bound
        self.bit_rate_higher_bound = bit_rate_higher_bound

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges() * self.bands, np.sum(self.num_spectrum_resources)), 
                                                 fill_value=-1, dtype=np.int) ###changed to array for diff bands

        # do we allow proactive rejection or not?
        self.reject_action = 1 if allow_rejection else 0

        # defining the observation and action spaces
        self.actions_output = np.zeros((self.k_paths + 1, self.bands +1,
                                       np.sum(self.num_spectrum_resources) + 1),
                                       dtype=int)
        self.episode_actions_output = np.zeros((self.k_paths + 1, self.bands+1,
                                               np.sum(self.num_spectrum_resources) + 1),
                                               dtype=int)
        self.actions_taken = np.zeros((self.k_paths + 1, self.bands +1,
                                      np.sum(self.num_spectrum_resources) + 1),
                                      dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + 1, self.bands + 1,
                                               np.sum(self.num_spectrum_resources) + 1),
                                              dtype=int)
        self.action_space = gym.spaces.MultiDiscrete((self.k_paths + self.reject_action, self.bands +self.reject_action,
                                                     np.sum(self.num_spectrum_resources) + self.reject_action))
        self.observation_space = gym.spaces.Dict(
            {'topology': gym.spaces.Discrete(10),
             'current_service': gym.spaces.Discrete(10)}
        )
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)

        self.logger = logging.getLogger('rmsaenv')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.warning(
                'Logging is enabled for DEBUG which generates a large number of messages. '
                'Set it to INFO if DEBUG is not necessary.')
        self._new_service = False
        if reset:
            self.reset(only_counters=False)
        

    def step(self, action: [int]):
        path, band, initial_slot = action[0], action[1], action[2]
        #print("Agent's Action: Path, Band, Initial Slot")
        #print(path, band, initial_slot)
        #time.sleep(0.05)
        self.band = band
        if self.band == 0:
            self.num_C_selected += 1
        elif self.band == 1:
            self.num_L_selected += 1
        elif self.band == 2:
            self.num_S_selected += 1
        elif self.band == 3:
            self.num_E_selected += 1
        
        if path == 0:
            self.num_path1_selected += 1
        elif path == 1:
            self.num_path2_selected += 1
        elif path == 2:
            self.num_path3_selected += 1
        elif path == 3:
            self.num_path4_selected += 1
        elif path == 4:
            self.num_path5_selected += 1
        
        if path != 5:
            self.total_path_length_per_episode += self.k_shortest_paths[self.service.source, self.service.destination][path].length
        self.actions_output[path, band, initial_slot] += 1 #
        if path < self.k_paths and band < self.bands and initial_slot < (np.sum(self.num_spectrum_resources)):  # action is for assigning a path
            slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path], band) #
            #print("Source and destination", self.service.source, self.service.destination)
            self.logger.debug('{} processing action {} path {} and initial slot {} for {} slots'.format(self.service.service_id, action, path, initial_slot, slots))
            if self.is_path_free(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                 initial_slot, slots, band):
                self._provision_path(self.k_shortest_paths[self.service.source, self.service.destination][path],
                                     initial_slot, slots, band)
                self.service.accepted = True
                self.actions_taken[path, band, initial_slot] += 1
                self._add_release(self.service)
            else:
                self.service.accepted = False
        else:
            self.service.accepted = False
        if not self.service.accepted:
            #print("===============================")
            #print(self.band)
            #print(self.k_paths, self.band, self.num_spectrum_resources[self.band])
            self.actions_taken[self.k_paths, self.bands, initial_slot] += 1
        #print(self.service.accepted)
        self.services_processed += 1
        self.episode_services_processed += 1
        self.bit_rate_requested += self.service.bit_rate
        self.episode_bit_rate_requested += self.service.bit_rate

        self.topology.graph['services'].append(self.service)

        reward = self.reward()
        info = {
                   'service_blocking_rate': (self.services_processed - self.services_accepted) / self.services_processed,
                   'episode_service_blocking_rate': (self.episode_services_processed - self.episode_services_accepted) / self.episode_services_processed,
                   'bit_rate_blocking_rate': (self.bit_rate_requested - self.bit_rate_provisioned) / self.bit_rate_requested,
                   'episode_bit_rate_blocking_rate': (self.episode_bit_rate_requested - self.episode_bit_rate_provisioned) / self.episode_bit_rate_requested,
                   'num_C_selected': self.num_C_selected,
                   'num_L_selected': self.num_L_selected,
                   'num_S_selected': self.num_S_selected,
                   'num_E_selected': self.num_E_selected,
                   'num_path1_selected': self.num_path1_selected,
                   'num_path2_selected': self.num_path2_selected,
                   'num_path3_selected': self.num_path3_selected,
                   'num_path4_selected': self.num_path4_selected,
                   'num_path5_selected': self.num_path5_selected,
                   'total_path_length_per_episode': self.total_path_length_per_episode
               }

        self._new_service = False
        self._next_service()
        
        self.blockingReason.to_csv('blockingReason{}.csv'.format(self.mean_service_holding_time*10))
        return self.observation(), reward, self.episode_services_processed == self.episode_length, info

    def reset(self, only_counters=True):
        self.episode_bit_rate_requested = 0
        self.episode_bit_rate_provisioned = 0
        self.episode_services_processed = 0
        self.episode_services_accepted = 0
        self.num_C_selected = 0
        self.num_L_selected = 0
        self.num_S_selected = 0
        self.num_E_selected = 0
        self.num_path1_selected = 0
        self.num_path2_selected = 0
        self.num_path3_selected = 0
        self.num_path4_selected = 0
        self.num_path5_selected = 0
        self.total_path_length_per_episode = 0
        self.episode_actions_output = np.zeros((self.k_paths + self.reject_action,
                                                self.bands + self.reject_action,
                                                np.sum(self.num_spectrum_resources) + self.reject_action),
                                               dtype=int)
        self.episode_actions_taken = np.zeros((self.k_paths + self.reject_action,
                                                self.bands + self.reject_action,
                                               np.sum(self.num_spectrum_resources) + self.reject_action),
                                              dtype=int)

        if only_counters:
            return self.observation()

        super().reset()

        self.bit_rate_requested = 0
        self.bit_rate_provisioned = 0

        self.topology.graph["available_slots"] = np.ones((self.topology.number_of_edges(), np.sum(self.num_spectrum_resources)), dtype=int) #including multiple bands

        self.spectrum_slots_allocation = np.full((self.topology.number_of_edges(), np.sum(self.num_spectrum_resources)),
                                                 fill_value=-1, dtype=np.int)

        self.topology.graph["compactness"] = 0.
        self.topology.graph["throughput"] = 0.
        for idx, lnk in enumerate(self.topology.edges()):
            self.topology[lnk[0]][lnk[1]]['external_fragmentation'] = 0.
            self.topology[lnk[0]][lnk[1]]['compactness'] = 0.

        self._new_service = False
        self._next_service()
        return self.observation()

    def render(self, mode='human'):
        return

    def _provision_path(self, path: Path, initial_slot, number_slots, idband): #No significant change other than where #
        # usage
        if not self.is_path_free(path, initial_slot, number_slots, idband):
            raise ValueError("Path {} has not enough capacity on slots {}-{}".format(path.node_list, path, initial_slot,
                                                                                     initial_slot + number_slots))

        self.logger.debug('{} assigning path {} on initial slot {} for {} slots'.format(self.service.service_id, path.node_list, initial_slot, number_slots))
        for i in range(len(path.node_list) - 1):
            self.topology.graph['available_slots'][self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                                                                        initial_slot:initial_slot + number_slots] = 0
            self.spectrum_slots_allocation[self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                                                    initial_slot:initial_slot + number_slots] = self.service.service_id
            self.topology[path.node_list[i]][path.node_list[i + 1]]['services'].append(self.service)
            self.topology[path.node_list[i]][path.node_list[i + 1]]['running_services'].append(self.service)
            self._update_link_stats(path.node_list[i], path.node_list[i + 1])
        self.topology.graph['running_services'].append(self.service)
        self.service.route = path
        self.service.band = idband #needed?
        self.service.initial_slot = initial_slot
        self.service.number_slots = number_slots
        self._update_network_stats()

        self.services_accepted += 1
        self.episode_services_accepted += 1
        self.bit_rate_provisioned += self.service.bit_rate
        self.episode_bit_rate_provisioned += self.service.bit_rate

    def _release_path(self, service: Service):  #No change?
        for i in range(len(service.route.node_list) - 1):
            self.topology.graph['available_slots'][
                self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'] ,
                                            service.initial_slot:service.initial_slot + service.number_slots] = 1
            self.spectrum_slots_allocation[self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['index'],
                                            service.initial_slot:service.initial_slot + service.number_slots] = -1
            self.topology[service.route.node_list[i]][service.route.node_list[i + 1]]['running_services'].remove(service)
            self._update_link_stats(service.route.node_list[i], service.route.node_list[i + 1])
        self.topology.graph['running_services'].remove(service)

    def _update_network_stats(self):
        last_update = self.topology.graph['last_update']
        time_diff = self.current_time - last_update
        if self.current_time > 0:
            last_throughput = self.topology.graph['throughput']
            last_compactness = self.topology.graph['compactness']

            cur_throughput = 0.

            for service in self.topology.graph["running_services"]:
                cur_throughput += service.bit_rate

            throughput = ((last_throughput * last_update) + (cur_throughput * time_diff)) / self.current_time
            self.topology.graph['throughput'] = throughput

            compactness = ((last_compactness * last_update) + (self._get_network_compactness() * time_diff)) / \
                              self.current_time
            self.topology.graph['compactness'] = compactness

        self.topology.graph['last_update'] = self.current_time

    def _update_link_stats(self, node1: str, node2: str):
        last_update = self.topology[node1][node2]['last_update']
        time_diff = self.current_time - self.topology[node1][node2]['last_update']
        if self.current_time > 0:
            last_util = self.topology[node1][node2]['utilization']   
            cur_util = ((np.sum(self.num_spectrum_resources)) - np.sum(  #Utilisation of single band or all bands? Currently did all
                self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :])) / \
                       (np.sum(self.num_spectrum_resources))
            utilization = ((last_util * last_update) + (cur_util * time_diff)) / self.current_time
            self.topology[node1][node2]['utilization'] = utilization

            slot_allocation = self.topology.graph['available_slots'][self.topology[node1][node2]['index'], :]

            # implementing fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
            last_external_fragmentation = self.topology[node1][node2]['external_fragmentation']
            last_compactness = self.topology[node1][node2]['compactness']

            cur_external_fragmentation = 0.
            cur_link_compactness = 0.
            if np.sum(slot_allocation) > 0:
                initial_indices, values, lengths = RMSAEnv1.rle(slot_allocation)

                # computing external fragmentation from https://ieeexplore.ieee.org/abstract/document/6421472
                unused_blocks = [i for i, x in enumerate(values) if x == 1]
                max_empty = 0
                if len(unused_blocks) > 1 and unused_blocks != [0, len(values) - 1]:
                    max_empty = max(lengths[unused_blocks])
                cur_external_fragmentation = 1. - (float(max_empty) / float(np.sum(slot_allocation)))

                # computing link spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6421472
                used_blocks = [i for i, x in enumerate(values) if x == 0]

                if len(used_blocks) > 1:
                    lambda_min = initial_indices[used_blocks[0]]
                    lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]

                    # evaluate again only the "used part" of the spectrum
                    internal_idx, internal_values, internal_lengths = RMSAEnv1.rle(
                        slot_allocation[lambda_min:lambda_max])
                    unused_spectrum_slots = np.sum(1 - internal_values)

                    if unused_spectrum_slots > 0:
                        cur_link_compactness = ((lambda_max - lambda_min) / np.sum(1 - slot_allocation)) * (
                                    1 / unused_spectrum_slots)
                    else:
                        cur_link_compactness = 1.
                else:
                    cur_link_compactness = 1.

            external_fragmentation = ((last_external_fragmentation * last_update) + (cur_external_fragmentation * time_diff)) / self.current_time
            self.topology[node1][node2]['external_fragmentation'] = external_fragmentation

            link_compactness = ((last_compactness * last_update) + (cur_link_compactness * time_diff)) / self.current_time
            self.topology[node1][node2]['compactness'] = link_compactness

        self.topology[node1][node2]['last_update'] = self.current_time

    def _next_service(self):
        if self._new_service:
            return
        at = self.current_time + self.rng.expovariate(1 / self.mean_service_inter_arrival_time)
        self.current_time = at

        ht = self.rng.expovariate(1 / self.mean_service_holding_time)
        src, src_id, dst, dst_id = self._get_node_pair()

        bit_rate = self.rng.randint(self.bit_rate_lower_bound, self.bit_rate_higher_bound)

        # release connections up to this point
        while len(self._events) > 0:
            (time, service_to_release) = heapq.heappop(self._events)
            if time <= self.current_time:
                self._release_path(service_to_release)
            else:  # release is not to be processed yet
                self._add_release(service_to_release)  # puts service back in the queue
                break  # breaks the loop

        self.service = Service(self.episode_services_processed, src, src_id,
                               destination=dst, destination_id=dst_id,
                               arrival_time=at, holding_time=ht, bit_rate=bit_rate)
        self._new_service = True

    def _get_path_slot_id(self, action: int) -> (int, int):
        """
        Decodes the single action index into the path index and the slot index to be used.

        :param action: the single action index
        :return: path index and initial slot index encoded in the action
        """
        path = int(action / self.num_spectrum_resources[self.band])
        initial_slot = action % self.num_spectrum_resources[self.band]
        return path, initial_slot

    def get_number_slots(self, path: Path, bnd) -> int:
        """
        Method that computes the number of spectrum slots necessary to accommodate the service request into the path.
        The method already adds the guardband.
        """
        if bnd == 0:
            return math.ceil(self.service.bit_rate / path.best_modulationC['capacity']) + 1
        elif bnd == 1:
            return math.ceil(self.service.bit_rate / path.best_modulationL['capacity']) + 1
        elif bnd ==2:
            return math.ceil(self.service.bit_rate / path.best_modulationS['capacity']) + 1
        elif bnd == 3:
            return math.ceil(self.service.bit_rate / path.best_modulationE['capacity']) + 1

    def is_path_free(self, path: Path, initial_slot: int, number_slots: int, idband) -> bool:
        if idband == 0:
            if initial_slot + number_slots > np.sum(self.num_spectrum_resources[:idband+1]):   #ensure not using multiple bands
                # logging.debug('error index' + env.parameters.rsa_algorithm)
                self.blockingReason = self.blockingReason.append(pd.DataFrame({'Source': [self.service.source], 'Destination':[self.service.destination], 'Reason': ['Band Overlap'], 'Path':[path]}), ignore_index = True)
                return False
        else:
            if initial_slot < np.sum(self.num_spectrum_resources[:idband]) or (initial_slot + number_slots) > np.sum(self.num_spectrum_resources[:idband+1]):   #ensure not using multiple bands
                # logging.debug('error index' + env.parameters.rsa_algorithm)
                self.blockingReason =  self.blockingReason.append(pd.DataFrame({'Source': [self.service.source], 'Destination':[self.service.destination], 'Reason': ['Band Overlap'], 'Path':[path]}), ignore_index = True)
                return False
        for i in range(len(path.node_list) - 1):
            if np.any(self.topology.graph['available_slots'][  #check
                      self.topology[path.node_list[i]][path.node_list[i + 1]]['index'],
                      initial_slot:initial_slot + number_slots] == 0):
                self.blockingReason = self.blockingReason.append(pd.DataFrame({'Source': [self.service.source], 'Destination':[self.service.destination], 'Reason': ['Slots full'], 'Path':[path], 'Nodes':[path.node_list[i],path.node_list[i + 1]]}), ignore_index = True)
                return False
        return True

    def get_available_slots(self, path: Path):
        available_slots = functools.reduce(np.multiply,
            self.topology.graph["available_slots"][[(self.topology[path.node_list[i]][path.node_list[i + 1]]['id'])
                                                    for i in range(len(path.node_list) - 1)], :])
        return available_slots

    def rle(inarray):
        """ run length encoding. Partial credit to R rle function.
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        # from: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return p, ia[i], z

    def get_available_blocks(self, path, idb): # mb included
        # get available slots across the whole path
        # 1 if slot is available across all the links
        # zero if not
        available_slots = self.get_available_slots(
            self.k_shortest_paths[self.service.source, self.service.destination][path])  #edit function
        #print(available_slots)
        if idb != 0:  #making slots of different band unavailable
          available_slots[:self.num_spectrum_resources[0]] = 0
        if idb != 1:  #making slots of different band unavailable
          available_slots[self.num_spectrum_resources[0]:(self.num_spectrum_resources[0]+self.num_spectrum_resources[1])] = 0
        if idb != 2:  #making slots of different band unavailable
          available_slots[(self.num_spectrum_resources[0]+self.num_spectrum_resources[1]):(self.num_spectrum_resources[0]+self.num_spectrum_resources[1]+self.num_spectrum_resources[2])] = 0
        if idb != 3:
          available_slots[(self.num_spectrum_resources[0]+self.num_spectrum_resources[1]+self.num_spectrum_resources[2]):] = 0
        #### add for final band
        # getting the number of slots necessary for this service across this path
        slots = self.get_number_slots(self.k_shortest_paths[self.service.source, self.service.destination][path], idb)
        # getting the blocks
        initial_indices, values, lengths = RMSAEnv1.rle(available_slots)
        #print("available", initial_indices)
        #print(initial_indices, values, lengths)
        # selecting the indices where the block is available, i.e., equals to one
        available_indices = np.where(values == 1)
        
        # selecting the indices where the block has sufficient slots
        sufficient_indices = np.where(lengths >= slots)
        #print(sufficient_indices)
        
        # getting the intersection, i.e., indices where the slots are available in sufficient quantity
        # and using only the J first indices
        final_indices = np.intersect1d(available_indices, sufficient_indices)[:self.j]
        return initial_indices[final_indices], lengths[final_indices]
          
          
        

    def _get_network_compactness(self):
        # implementing network spectrum compactness from https://ieeexplore.ieee.org/abstract/document/6476152

        sum_slots_paths = 0  # this accounts for the sum of all Bi * Hi

        for service in self.topology.graph["running_services"]:
            sum_slots_paths += service.number_slots * service.route.hops

        # this accounts for the sum of used blocks, i.e.,
        # \sum_{j=1}^{M} (\lambda_{max}^j - \lambda_{min}^j)
        sum_occupied = 0

        # this accounts for the number of unused blocks \sum_{j=1}^{M} K_j
        sum_unused_spectrum_blocks = 0

        for n1, n2 in self.topology.edges():
            # getting the blocks
            initial_indices, values, lengths = \
                RMSAEnv1.rle(self.topology.graph['available_slots'][self.topology[n1][n2]['index'], :])
            used_blocks = [i for i, x in enumerate(values) if x == 0]
            if len(used_blocks) > 1:
                lambda_min = initial_indices[used_blocks[0]]
                lambda_max = initial_indices[used_blocks[-1]] + lengths[used_blocks[-1]]
                sum_occupied += lambda_max - lambda_min  # we do not put the "+1" because we use zero-indexed arrays

                # evaluate again only the "used part" of the spectrum
                internal_idx, internal_values, internal_lengths = RMSAEnv1.rle(
                    self.topology.graph['available_slots'][self.topology[n1][n2]['index'], lambda_min:lambda_max])
                sum_unused_spectrum_blocks += np.sum(internal_values)

        if sum_unused_spectrum_blocks > 0:
            cur_spectrum_compactness = (sum_occupied / sum_slots_paths) * (self.topology.number_of_edges() /
                                                                           sum_unused_spectrum_blocks)
        else:
            cur_spectrum_compactness = 1.

        return cur_spectrum_compactness

### Removed function below as never called