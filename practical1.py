import numpy as np
import random
import csv
import os
import matplotlib



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

def evolve_population(population, generations: int, fitness_func, crossover,csv_filename):
    no_improve_count = 0
    global_optimum = "1" * 40
    optimum_fitness = fitness_func(global_optimum)

    # Use 'with' to open the file and keep all writes inside
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Generation", "Population Size", "Crossover Used", "Best Fitness", "Mean Fitness"])

        for gen in range(generations):
            population = np.random.permutation(population)  # Shuffle 
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

                # If the global optimum is reached
                if child1 == global_optimum or child2 == global_optimum:
                    best1, best2 = global_optimum, global_optimum
                    best1_fitness, best2_fitness = optimum_fitness, optimum_fitness
                    new_population[i] = (best1, best1_fitness)
                    new_population[i + 1] = (best2, best2_fitness)
                    population = new_population

                    # Log the final generation stats
                    best_solution = max(population, key=lambda x: x['fitness'])
                    mean_fitness = np.mean([ind['fitness'] for ind in population])
                    writer.writerow([gen, len(population), crossover.__name__, best_solution['fitness'], mean_fitness])

                    return population

                # Family competition: pick best two of parent1, parent2, child1, child2
                sorted_indices = np.argsort(fitness_values)[::-1]  # sort descending
                best1, best2 = candidates[sorted_indices[0]], candidates[sorted_indices[1]]
                best1_fitness, best2_fitness = fitness_values[sorted_indices[0]], fitness_values[sorted_indices[1]]

                new_population[i] = (best1, best1_fitness)
                new_population[i + 1] = (best2, best2_fitness)

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= 20:
                # Log stats before returning
                best_solution = max(new_population, key=lambda x: x['fitness'])
                mean_fitness = np.mean([ind['fitness'] for ind in new_population])
                writer.writerow([gen, len(new_population), crossover.__name__, best_solution['fitness'], mean_fitness])
                return new_population

            population = new_population  # Update current population

            # Log the stats for this generation
            best_solution = max(population, key=lambda x: x['fitness'])
            mean_fitness = np.mean([ind['fitness'] for ind in population])
            writer.writerow([gen, len(population), crossover.__name__, best_solution['fitness'], mean_fitness])

    return population

''' 
todo : Add loose linkage function for scenario 4, should be easy just add an offset to the loop when creating k blocks
'''

if __name__ == "__main__":
    config = {
        'pop_size': 180,
        'generations': 50,
        'fitness_func': non_deceptive_trap_function,       # or deceptive_trap_function
        'crossover': uniform_crossover     # or uniform_crossover
    }

# Batch running

    if config['crossover'] == two_point_crossover:
        folder_name = "2_points"
    else:
        folder_name = "Uniform"

    # Create the folder if it does not exist (to avoid errors)
    # Store them under "Experiments/Experiment{i}"
    base_folder = "Experiments/Experiment3"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Run 10 experiments
    for i in range(1, 11):
        # Init
        population = init_pop_generation(config['pop_size'], fitness_func=config['fitness_func'])

        # Build filename
        csv_filename = os.path.join(output_folder, f"experiment1-{i}.csv")

        # Run the Alg and save results
        final_population = evolve_population(
            population, 
            config['generations'], 
            fitness_func=config['fitness_func'],
            crossover=config['crossover'],
            csv_filename=csv_filename
        )

 
        best_solution = max(final_population, key=lambda x: x['fitness'])
        mean_fitness = np.mean([ind['fitness'] for ind in final_population])
        print(f"\n--- Run {i} Finished ---")
        print(f"CSV saved to: {csv_filename}")
        print("Best Solution:", best_solution)
        print("Mean Fitness:", mean_fitness)
