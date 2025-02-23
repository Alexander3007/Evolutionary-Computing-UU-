import numpy as np
import random

def init_pop_generation(size: int, fitness_func):
    population = np.array([''.join(random.choices('01', k=40)) for _ in range(size)])
    fitness_scores = np.array([fitness_func(s) for s in population]) 
    structured_population = np.array(list(zip(population, fitness_scores)), dtype=[('individual', 'U40'), ('fitness', float)])
    return structured_population

def two_point_crossover(parent1: str, parent2: str):
    point1, point2 = sorted(random.sample(range(1, 39), 2))  # Ensure point1 < point2
    offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return offspring1, offspring2

def uniform_crossover(parent1: str, parent2: str):
    child1 = ''.join(p1 if random.random() > 0.5 else p2 for p1, p2 in zip(parent1, parent2))
    child2 = ''.join(p2 if random.random() > 0.5 else p1 for p1, p2 in zip(parent1, parent2))
    return child1, child2

def non_deceptive_trap_function(solution: str):
    block_list = [solution[i:i+4] for i in range(0, 40, 4)]
    fitness_score = 0
    for blocks in block_list:
        block_fit = blocks.count('1') 
        if block_fit == 4:
            fitness_score += 4 
        elif block_fit == 2:
            fitness_score += 0.5
        elif block_fit == 1:
            fitness_score += 1
        elif block_fit == 0:
            fitness_score += 1.5
    return fitness_score

def counting_ones(solution: str):
    return solution.count('1')

def deceptive_trap_function(solution: str):
    block_list = [solution[i:i+4] for i in range(0, 40, 4)]
    fitness_score = 0
    for blocks in block_list:
        block_fit = blocks.count('1') 
        if block_fit == 4:
            fitness_score += 4 
        elif block_fit == 2:
            fitness_score += 1
        elif block_fit == 1:
            fitness_score += 2
        elif block_fit == 0:
            fitness_score += 3
    return fitness_score

def evolve_population(population, generations: int, fitness_func, crossover):
    no_improve_count = 0
    global_optimum = "1" * 40
    optimum_fitness = fitness_func(global_optimum)
    for gen in range(generations):
        population = np.random.permutation(population)  # Shuffle the population
        new_population = population.copy()
        improved = False
        
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i]['individual'], population[i+1]['individual']
            child1, child2 = crossover(parent1, parent2)
            
            # Evaluate fitness
            candidates = [parent1, parent2, child1, child2]
            fitness_values = np.array([fitness_func(s) for s in candidates])
            
            if max(fitness_values[2:]) > max(fitness_values[:2]):
                improved = True
            
            if child1 == global_optimum or child2 == global_optimum:
                best1, best2 = global_optimum, global_optimum
                best1_fitness, best2_fitness = optimum_fitness, optimum_fitness
                new_population[i] = (best1, best1_fitness)
                new_population[i + 1] = (best2, best2_fitness)
                population = new_population
                return population
            
            # Select best two individuals
            sorted_indices = np.argsort(fitness_values)[::-1]  # Sort descending
            best1, best2 = candidates[sorted_indices[0]], candidates[sorted_indices[1]]
            best1_fitness, best2_fitness = fitness_values[sorted_indices[0]], fitness_values[sorted_indices[1]]

            # Update new population
            new_population[i] = (best1, best1_fitness)
            new_population[i + 1] = (best2, best2_fitness)

        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0

        if no_improve_count >= 20:  # Stop if no improvement in last 20 generations
            return new_population

        population = new_population  # Update current population
    
    return population

''' 
todo : Add loose linkage function for scenario 4, should be easy just add an offset to the loop when creating k blocks
'''

# Define the configuration
config = {
    'pop_size': 320,
    'generations': 50,
    'fitness_func': non_deceptive_trap_function,  # Choose from: non_deceptive_trap_function, deceptive_trap_function, counting_ones
    'crossover': uniform_crossover # or uniform_crossover
}

# Initialize population using the config
population = init_pop_generation(config['pop_size'], fitness_func=config['fitness_func'])

# Evolve population using the config
final_population = evolve_population(
    population, 
    config['generations'], 
    fitness_func=config['fitness_func'],
    crossover=config['crossover']
)

best_solution = max(final_population, key=lambda x: x['fitness'])
mean_fitness = np.mean([ind['fitness'] for ind in final_population])

print("\n--- Final Population ---")
print(final_population)
print("\nBest Solution:", best_solution)
print("Mean Fitness:", mean_fitness)