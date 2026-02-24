import random
import math
from pm4py.objects.petri_net import semantics
from tqdm.notebook import tqdm

class AutomataWrapper:
    """
    Wrapper class for the Petri net.
    """
    def __init__(self, net, initial_marking, final_marking, perfect_alignments):
        """
        Initializes the Petri net wrapper.

        Args:
            net: The Petri net.
            initial_marking: The initial marking of the Petri net.
            final_marking: The final marking of the Petri net.
            perfect_alignments: The perfect alignments of the Petri net.
        """
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.perfect_alignments = perfect_alignments
        
        self.current_marking = None
        self.current_index = None
        self.current_alignment = None
        self.reset()
    
    def reset(self):
        """
        Resets the Petri net to the initial marking.
        """
        self.current_marking = self.initial_marking.copy()
        self.current_index = 0
        self.current_alignment = random.choice(self.perfect_alignments)['alignment']
    
    def step(self, transition_label):
        """
        Executes a transition in the Petri net.

        Args:
            transition_label: The label of the transition to execute.

        Returns:
            The transition and the next marking.
        """
        if self.current_index >= len(self.current_alignment):
            return None, self.final_marking

        # Get enabled transitions from the current marking in the Petri net
        trs = semantics.enabled_transitions(self.net, self.current_marking)
        transition = None
        for t in trs:
            if t.name == transition_label:
                transition = t
                break
                
        if transition is None:
            raise Exception(f"Transition {transition_label} not enabled in current marking {self.current_marking}")
        
        self.current_marking = semantics.execute(transition, self.net, self.current_marking)
        self.current_index += 1
        
        return transition, self.current_marking

    def generate_mdp(self, aligned_log):  
        """
        Generates the MDP from the Petri net and the aligned log.

        Args:
            aligned_log: The aligned log.

        Returns:
            The states of the MDP.
        """
        Q0 = State(self.initial_marking, index=0) # Initial state
        
        states = {}
        states[(str(self.initial_marking), 0)] = Q0
        
        print(f"Generating MDP states with {len(aligned_log)} traces...")

        for i, (alignment, trace) in enumerate(zip(tqdm(self.perfect_alignments), aligned_log)):
            self.reset()
            
            current_state = Q0
            self.current_alignment = alignment['alignment']
            
            for step in alignment['alignment']:
                # Get the transition label
                transition_label = step[0][1]
                # Execute the transition
                transition, next_marking = self.step(transition_label)
                # Get the next index
                next_index = self.current_index
                
                # Get the state key
                state_key = (str(next_marking), next_index)
                
                # If the state is not in the dictionary, add it
                if state_key not in states:
                    states[state_key] = State(next_marking, next_index)
                
                # Get the next state
                next_state = states[state_key]
                
                # Add the trace to the current state
                current_state.add_trace(i, trace)
                
                # Record the transition
                if transition is not None:
                    current_state.record_transition(transition, next_marking, next_index)
                
                # Move to the next state
                current_state = next_state
                
            # Add the trace to the final state
            current_state.add_trace(i, trace)
                
        print(f"MDP generated with {len(states)} states.")
        return states
    
class State:
    """
    Represents a state in the MDP.
    """
    def __init__(self, marking, index):
        """
        Initializes the state.

        Args:
            marking: The marking of the state.
            index: The index of the state.
        """
        self.marking = marking
        self.index = index
        self.domain = {}
        self.outgoing_transitions = {}
        
    def add_trace(self, trace_id, trace):
        """
        Adds a trace to the state.

        Args:
            trace_id: The index of the trace.
            trace: The trace to add.
        """
        self.domain[trace_id] = trace
        
    def record_transition(self, transition_name, next_marking, next_index):
        """
        Records a transition from the current state to the next state.

        Args:
            transition_name: The name of the transition.
            next_marking: The marking of the next state.
            next_index: The index of the next state.
        """
        key = (transition_name, next_marking, next_index)
        if key not in self.outgoing_transitions:
            self.outgoing_transitions[key] = 0     
        self.outgoing_transitions[key] += 1
        
    def get_transition_probabilities(self):
        """
        Calculates the probabilities of transitioning from the current state to the next state.

        Returns:
            A dictionary of transition probabilities.
        """
        probabilities = {}
            
        for (trans_name, next_mark, _), count in self.outgoing_transitions.items():
            prob = count / len(self)
            probabilities[f"{trans_name} -> {next_mark}"] = prob
            
        if len(probabilities) != 0:
            assert abs(sum(probabilities.values()) - 1.0) < 1e-6, "Probabilities do not sum to 1."
                
        return probabilities

    def calculate_state_entropy(self):
        """
        Calculates the entropy of the current state.

        Returns:
            The entropy of the current state.
        """
        entropy = 0.0
        probabilities = self.get_transition_probabilities()
        
        for prob in probabilities.values():
            entropy -= prob * math.log2(prob)
            
        return entropy

    def calculate_probability_to_reach_state(self, aligned_log):
        """
        Calculates the probability of reaching the current state from the initial state.

        Args:
            aligned_log: The aligned log.

        Returns:
            The probability of reaching the current state.
        """
        total_log_volume = sum(len(trace) for trace in aligned_log)
        
        return len(self) / total_log_volume
    
    def build_dataset(self, perfect_alignment):
        """
        Builds a dataset associated to the current state.

        Args:
            perfect_alignment: The perfect alignment.

        Returns:
            A dataset associated to the current state.
        """
        dataset = []
        state_index = self.index
        
        if len(self.outgoing_transitions) != 0:
            for i, trace in self.domain.items():
                alignment = perfect_alignment[i]['alignment']
                event_index = 0
                x = []
                
                for j in range(state_index):
                    transition = alignment[j][0][0]
                    # If the transition is not a skip, add the event to the dataset
                    if transition != '>>':
                        x.append(trace[event_index])
                        event_index +=1
                
                # Get the next transition
                y = alignment[state_index][0][1]
                
                # Add the dataset
                dataset.append((x, y))
            
        return dataset
                
    def __len__(self):
        """
        Returns the number of traces in the current state.
        """
        return len(list(self.domain.keys()))
    
    def __str__(self):
        """
        Returns a string representation of the current state.
        """
        probabilities = self.get_transition_probabilities()
        prob_str = '\n\t'.join([f"{k}: {v:.4f}" for k, v in probabilities.items()])
        return f"State: ({self.marking}, {self.index})\n  - Domain size: {len(self)}\n  - Outgoing transitions: \n\t{prob_str}\n  - Entropy: {self.calculate_state_entropy():.4f}\n"