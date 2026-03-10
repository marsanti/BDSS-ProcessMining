from tqdm.notebook import tqdm

class LocalFormula:
    """
    Represents a local formula.
    """
    def __init__(self, check_function, description):
        """
        Initializes the local formula.

        Args:
            check_function: The function to check.
            description: The description of the formula.
        """
        # The logic: takes a vector, returns boolean
        self.evaluate = check_function 
        # For readability (e.g., "p1 AND NOT p5")
        self.description = description

    def __call__(self, vector):
        """
        Calls the local formula.

        Args:
            vector: The vector to check.

        Returns:
            The result of the check.
        """
        return self.evaluate(vector)

# --- HELPER FUNCTIONS TO BUILD FORMULAS ---

def Literal(function, description):
    """
    Factory for a simple literal p_h

    Args:
        function: The function to check.
        description: The description of the formula.

    Returns:
        The local formula.
    """
    return LocalFormula(
        function, 
        description
    )

def Not(formula):
    """
    Factory for Negation (NOT psi)

    Args:
        formula: The formula to negate.

    Returns:
        The negated formula.
    """
    return LocalFormula(
        lambda e: not formula(e), 
        f"NOT({formula.description})"
    )

def And(formula1, formula2):
    """
    Factory for Conjunction (psi1 AND psi2)

    Args:
        formula1: The first formula.
        formula2: The second formula.

    Returns:
        The conjunction of the two formulas.
    """
    return LocalFormula(
        lambda e: formula1(e) and formula2(e), 
        f"({formula1.description} AND {formula2.description})"
    )

class TraceTest:
    """
    Represents a trace test.
    """
    def __init__(self, description, psi_function: callable, time: float=None):
        """
        Initializes the trace test.

        Args:
            description: The description of the trace test.
            psi_function: The function to check.
            time: The time to check.
        """
        self.description = description
        self.psi = psi_function
        self.ts = time
        
    def check(self, x, index: int=None):
        """
        Checks if the trace test is satisfied.

        Args:
            x: The trace.
            index: The index of the trace.

        Returns:
            True if the trace test is satisfied, False otherwise.
        """
        if len(x) == 0:
            return False
        
        if index is not None:
            last_event = x[index]
        else:
            last_event = x[-1]
        last_time = last_event.get('time:timestamp')
        
        for event in x:
            past_event = event
            
            if self.psi(past_event):
                if self.ts is None:
                    return True
                else:
                    if index is None and event == x[-1]:
                        break

                    past_time = past_event.get('time:timestamp')
 
                    if last_time and past_time:
                        diff = (last_time - past_time).total_seconds()
                        
                        if diff <= self.ts:
                            return True
                   
        return False
        
    def partition(self, dataset):
        """
        Partitions the dataset into satisfied and unsatisfied sets.

        Args:
            dataset: The dataset to partition.

        Returns:
            The satisfied set and the unsatisfied set.
        """
        satisfied_set = []
        unsatisfied_set = []
        
        for x, y in dataset:      
            is_satisfied = self.check(x)
            if is_satisfied:
                satisfied_set.append(y)
            else:
                unsatisfied_set.append(y)
                
        return satisfied_set, unsatisfied_set

    def generate_state_datasets(self, states, perfect_alignment):
        """
        Generates the datasets for each state.

        Args:
            states: The states of the MDP.
            perfect_alignment: The perfect alignment.

        Returns:
            The overall dataset containing the satisfied and unsatisfied sets for each state.
        """
        state_datasets = {}

        print(f'Trace test for {self.description}...\n')
        for state in tqdm(sorted(states.values(), key=lambda n: n.index)):
            dataset = state.build_dataset(perfect_alignment)
            sat_set, unsat_set = self.partition(dataset)
            state_datasets[(str(state.marking), state.index)] = (dataset, sat_set, unsat_set)

        print(f'Datasets associated to each state computed.')
        
        return state_datasets