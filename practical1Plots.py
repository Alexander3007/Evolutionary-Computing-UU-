import numpy as np
import random
import csv
import os
import plotly.graph_objects as go
import time

# Global variable
fitness_evaluation_count = 0

def init_pop_generation(size: int, fitness_func):
    population = np.array([''.join(random.choices('01', k=40)) for _ in range(size)])
    fitness_scores = np.array([fitness_func(s) for s in population]) 
    structured_population = np.array(list(zip(population, fitness_scores)), dtype=[('individual', 'U40'), ('fitness', float)])
    return structured_population

def uniform_crossover(parent1: str, parent2: str):
    child1 = ''.join(p1 if random.random() > 0.5 else p2 for p1, p2 in zip(parent1, parent2))
    child2 = ''.join(p2 if random.random() > 0.5 else p1 for p1, p2 in zip(parent1, parent2))
    return child1, child2

def counting_ones(solution: str):
    global fitness_evaluation_count
    fitness_evaluation_count += 1
    return solution.count('1')

def evolve_population(population, generations: int, fitness_func, crossover):
    prop_t = []
    # best_fitness_t = []
    # avg_fitness_t = []
    # min_fitness_t = []
    
    for gen in range(generations):
        population = np.random.permutation(population)  # Shuffle 
        new_population = population.copy()
        fitness_values = []
        
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i]['individual'], population[i+1]['individual']
            child1, child2 = crossover(parent1, parent2)
            
            # Evaluate fitness
            candidates = [parent1, parent2, child1, child2]
            candidate_fitness = np.array([fitness_func(s) for s in candidates])
            
            # Family competition: pick best two of parent1, parent2, child1, child2
            sorted_indices = np.argsort(candidate_fitness)[::-1]  # sort descending
            best1, best2 = candidates[sorted_indices[0]], candidates[sorted_indices[1]]
            best1_fitness, best2_fitness = candidate_fitness[sorted_indices[0]], candidate_fitness[sorted_indices[1]]
            
            new_population[i] = (best1, best1_fitness)
            new_population[i + 1] = (best2, best2_fitness)
            fitness_values.append(best1_fitness)
            fitness_values.append(best2_fitness)
        
        # Correct proportion calculation
        total_bits_1 = sum(ind['individual'].count('1') for ind in population)
        proportion = total_bits_1 / (len(population) * 40)
        prop_t.append(proportion)
        
        # Print proportion at each generation
        print(f"Generation {gen}: Proportion of 1s = {proportion:.4f}")
        
        # best_fitness_t.append(np.max(fitness_values))
        # avg_fitness_t.append(np.mean(fitness_values))
        # min_fitness_t.append(np.min(fitness_values))
        
        population = new_population
        
        if proportion == 1.0:
            break
    
    return population, prop_t

if __name__ == "__main__":
    config = {
        'pop_size': 200,
        'generations': 100,
        'fitness_func': counting_ones,
        'crossover': uniform_crossover
    }
    
    start_time = time.time()
    population = init_pop_generation(config['pop_size'], fitness_func=config['fitness_func'])
    final_population, prop_t = evolve_population(
        population, config['generations'], fitness_func=config['fitness_func'], crossover=config['crossover']
    )
    elapsed_time = time.time() - start_time
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(prop_t))), y=prop_t, mode='lines', name='Proportion of 1-bits'))
    
    fig.update_layout(title='Proportion of 1-bits over Generations',
                      xaxis_title='Generation',
                      yaxis_title='Proportion of 1s',
                      template='plotly_white')
    
    # Save as PNG
    fig.write_image("ga_proportion_plot.png")
    
    print(f"Total runtime: {elapsed_time:.4f} seconds")
    print("Plot saved as ga_proportion_plot.png")