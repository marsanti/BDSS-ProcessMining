from pm4py.objects.log.obj import EventLog, Trace, Event
from collections import Counter
from tqdm.notebook import tqdm
import math
import copy

    
def compute_aligned_log(log, standard_alignments):
    """
    Computes alignments of the log against the Petri net.

    Args:
        log: The log.
        standard_alignments: The standard alignments.

    Returns:
        The aligned log.
    """
    aligned_log = EventLog()

    for alignment, log_trace in zip(standard_alignments, log):
        aligned_trace = Trace()
        
        index = 0
        
        last_timestamp = None
        if len(log_trace) > 0 and 'time:timestamp' in log_trace[0]:
            last_timestamp = log_trace[0]['time:timestamp']
            
        for transition in alignment['alignment']:
            a_i, b_i = transition[1]
            log_event = None
            
            # If the transition is a log event
            if a_i != '>>':
                log_event = log_trace[index]
                index += 1
                
                if 'time:timestamp' in log_event:
                    last_timestamp = log_event['time:timestamp']
            
            # If the transition is a model event
            if b_i != '>>' and b_i is not None:                
                # If the transition is a log event, copy the data from the log event
                if log_event is not None:
                    new_data = {'concept:name': b_i}
                    for key, value in log_event.items():
                        if key != 'concept:name':
                            new_data[key] = value
                else:
                    new_data = {'concept:name': f'dummy_{b_i}'}
                    if last_timestamp is not None:
                        new_data['time:timestamp'] = last_timestamp
                        
                event = Event(new_data)
                aligned_trace.append(event)

        aligned_log.append(aligned_trace)
    print(f"Aligned Log built with {len(aligned_log)} traces.")
    
    return aligned_log

def calculate_process_entropy(states, aligned_log):
    """
    Calculates the process entropy.

    Args:
        states: The states of the MDP.
        aligned_log: The aligned log.

    Returns:
        The process entropy.
    """
    process_entropy = 0.0
    
    for state in states.values():
        state_entropy = state.calculate_state_entropy()
        prob_to_reach_state = state.calculate_probability_to_reach_state(aligned_log)

        process_entropy += prob_to_reach_state * state_entropy
        
    return process_entropy

def calculate_subset_entropy(set):
    """
    Calculates the subset entropy.

    Args:
        set: The set.

    Returns:
        The subset entropy.
    """
    total = len(set)
    
    if total == 0:
        return 0.0
    
    counter = Counter(set)
    entropy = 0.0
    
    for count in counter.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    
    return entropy  

def calculate_trace_test_entropy(state_datasets, states, aligned_log):
    """
    Calculates the trace test entropy.

    Args:
        state_datasets: The datasets for each state.
        states: The states of the MDP.
        aligned_log: The aligned log.

    Returns:
        The trace test entropy.
    """
    entropy = 0.0

    for key, (dataset, sat_set, unsat_set) in state_datasets.items():
        if len(dataset) == 0:
            continue
        prob_to_reach_state = states[key].calculate_probability_to_reach_state(aligned_log)
        
        sat_proportion = len(sat_set) / len(dataset)
        unsat_proportion = len(unsat_set) / len(dataset)

        sat_entropy = calculate_subset_entropy(sat_set)
        unsat_entropy = calculate_subset_entropy(unsat_set)

        entropy += prob_to_reach_state * (sat_proportion * sat_entropy + unsat_proportion * unsat_entropy)

    return entropy

def generate_relabeled_log(original_log, test_set):
    """
    Relabels the log with the trace test results.

    Args:
        original_log: The original log.
        test_set: The set of trace tests.

    Returns:
        The relabeled log.
    """
    print(f"Relabeling log with {len(test_set)} tests...")
    new_log = copy.deepcopy(original_log)
    
    for trace in tqdm(new_log):
        for i, event in enumerate(trace):
            results_vector = []
            for test in test_set:
                is_satified = test.check(trace, i)
                results_vector.append(1 if is_satified else 0)

            vector_str = str(results_vector).replace(" ", "")
            original_name = event['concept:name']
            new_name = f"{original_name}_{vector_str}"
            event['concept:name'] = new_name
        
    print(f"Success! Relabeling completed.")
    return new_log