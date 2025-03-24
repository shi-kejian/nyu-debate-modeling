# train/prover_verifier_trainer.py
from train.base_trainer import BaseTrainer
from typing import List, Optional

class ProverVerifierTrainer(BaseTrainer):
    def __init__(
        self,
        prover: Prover,
        verifier: Verifier,
        config: ProverVerifierConfig
    ):
        self.prover = prover
        self.verifier = verifier
        self.config = config
        
    def train_iteration(self, problems: List[str]):
        solutions = []
        for problem in problems:
            solution = self.prover.generate_solution(problem)
            solutions.append(solution)
            
        scores = []
        for problem, solution in zip(problems, solutions):
            score, _ = self.verifier.verify_solution(problem, solution)
            scores.append(score)
            
        if self.prover.role == ProverRole.HELPFUL:
            self._update_helpful_prover(problems, solutions, scores)
        else:
            self._update_sneaky_prover(problems, solutions, scores)