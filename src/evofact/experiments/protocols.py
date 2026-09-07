from evofact.routing.strategies import STRATEGIES

class ExperimentProtocol:
    def __init__(self,manifest_id:str,arm:str):
        if arm not in STRATEGIES: raise ValueError(f"unknown arm: {arm}")
        self.manifest_id=manifest_id; self.arm=arm

def all_protocols(manifest_id:str)->list[ExperimentProtocol]: return [ExperimentProtocol(manifest_id,x) for x in STRATEGIES]
