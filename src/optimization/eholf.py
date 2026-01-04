"""
EHOLF (Evolved Hierarchical Optimization of Learned Features) hyperparameter optimizer.

This module implements a hierarchical evolutionary optimization algorithm for
hyperparameter tuning of the LSTM emotion classifier.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any
import copy


class EHOLF:
    """
    EHOLF hyperparameter optimizer using evolutionary strategies.
    
    This optimizer uses a hierarchical evolutionary approach to optimize
    hyperparameters by evolving a population of configurations.
    
    Args:
        param_space (dict): Dictionary defining the hyperparameter search space
        population_size (int): Size of the population
        generations (int): Number of generations to evolve
        mutation_rate (float): Probability of mutation
        crossover_rate (float): Probability of crossover
        elite_size (int): Number of elite individuals to preserve
        random_state (int): Random seed for reproducibility
    """
    
    def __init__(
        self,
        param_space: Dict[str, Dict[str, Any]],
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        random_state: int = 42
    ):
        self.param_space = param_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.history = []
        
    def _initialize_population(self) -> List[Dict]:
        """Initialize random population of hyperparameter configurations."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, param_config in self.param_space.items():
                param_type = param_config['type']
                
                if param_type == 'int':
                    individual[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
                elif param_type == 'float':
                    individual[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_type == 'categorical':
                    individual[param_name] = np.random.choice(
                        param_config['values']
                    )
                elif param_type == 'log':
                    # Log-scale sampling for parameters like learning rate
                    log_min = np.log10(param_config['min'])
                    log_max = np.log10(param_config['max'])
                    individual[param_name] = 10 ** np.random.uniform(log_min, log_max)
            
            # Convert numpy types to Python native types
            individual = self._convert_to_native_types(individual)
            population.append(individual)
        
        return population
    
    def _convert_to_native_types(self, individual: Dict) -> Dict:
        """Convert numpy types to Python native types."""
        converted = {}
        for param_name, value in individual.items():
            param_config = self.param_space[param_name]
            param_type = param_config['type']
            
            if param_type == 'int':
                converted[param_name] = int(value)
            elif param_type == 'float' or param_type == 'log':
                converted[param_name] = float(value)
            elif param_type == 'categorical':
                # Ensure categorical values are Python native types
                if isinstance(value, np.integer):
                    converted[param_name] = int(value)
                elif isinstance(value, np.floating):
                    converted[param_name] = float(value)
                else:
                    converted[param_name] = value
            else:
                converted[param_name] = value
        
        return converted
    
    def _mutate(self, individual: Dict) -> Dict:
        """Apply mutation to an individual."""
        mutated = copy.deepcopy(individual)
        
        for param_name, param_config in self.param_space.items():
            if np.random.random() < self.mutation_rate:
                param_type = param_config['type']
                
                if param_type == 'int':
                    # Gaussian mutation with bounds
                    std = (param_config['max'] - param_config['min']) * 0.1
                    mutated[param_name] = int(np.clip(
                        mutated[param_name] + np.random.normal(0, std),
                        param_config['min'],
                        param_config['max']
                    ))
                elif param_type == 'float':
                    # Gaussian mutation with bounds
                    std = (param_config['max'] - param_config['min']) * 0.1
                    mutated[param_name] = np.clip(
                        mutated[param_name] + np.random.normal(0, std),
                        param_config['min'],
                        param_config['max']
                    )
                elif param_type == 'categorical':
                    mutated[param_name] = np.random.choice(
                        param_config['values']
                    )
                elif param_type == 'log':
                    # Log-scale mutation
                    log_val = np.log10(mutated[param_name])
                    log_min = np.log10(param_config['min'])
                    log_max = np.log10(param_config['max'])
                    log_std = (log_max - log_min) * 0.1
                    new_log_val = np.clip(
                        log_val + np.random.normal(0, log_std),
                        log_min,
                        log_max
                    )
                    mutated[param_name] = 10 ** new_log_val
        
        return self._convert_to_native_types(mutated)
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents."""
        if np.random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = {}
        child2 = {}
        
        for param_name in self.param_space.keys():
            if np.random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return self._convert_to_native_types(child1), self._convert_to_native_types(child2)
    
    def _select_parents(self, population: List[Dict], fitness_scores: List[float]) -> Tuple[Dict, Dict]:
        """Select two parents using tournament selection."""
        tournament_size = 3
        
        def tournament():
            indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            return population[winner_idx]
        
        return tournament(), tournament()
    
    def optimize(
        self,
        objective_function: Callable[[Dict], float],
        verbose: bool = True
    ) -> Tuple[Dict, float]:
        """
        Optimize hyperparameters using EHOLF algorithm.
        
        Args:
            objective_function: Function that takes hyperparameters and returns fitness score
            verbose: Whether to print progress
        
        Returns:
            Tuple of (best_hyperparameters, best_fitness_score)
        """
        # Initialize population
        self.population = self._initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            self.fitness_scores = []
            for individual in self.population:
                fitness = objective_function(individual)
                self.fitness_scores.append(fitness)
            
            # Track best individual
            max_fitness_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[max_fitness_idx] > self.best_fitness:
                self.best_fitness = self.fitness_scores[max_fitness_idx]
                self.best_individual = copy.deepcopy(self.population[max_fitness_idx])
            
            # Store history
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(self.fitness_scores),
                'std_fitness': np.std(self.fitness_scores)
            })
            
            if verbose:
                print(f"Generation {generation + 1}/{self.generations}: "
                      f"Best Fitness = {self.best_fitness:.4f}, "
                      f"Mean Fitness = {np.mean(self.fitness_scores):.4f}")
            
            # Create next generation
            # 1. Preserve elite individuals
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            elite_individuals = [self.population[i] for i in sorted_indices[:self.elite_size]]
            
            # 2. Generate offspring
            offspring = []
            while len(offspring) < self.population_size - self.elite_size:
                parent1, parent2 = self._select_parents(self.population, self.fitness_scores)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                offspring.extend([child1, child2])
            
            # Combine elite and offspring for next generation
            self.population = elite_individuals + offspring[:self.population_size - self.elite_size]
        
        return self.best_individual, self.best_fitness
    
    def get_optimization_history(self) -> List[Dict]:
        """Get the optimization history."""
        return self.history
