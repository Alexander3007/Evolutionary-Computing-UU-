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
    errors_t = []    # Selection errors 
    correct_t = []   # Correct selection decisions
    
    # Initialize metrics lists
    schema1_count = []   # individuals with first bit '1'
    schema0_count = []   # individuals with first bit '0'
    schema1_avg = []     # Average fitness for schema 1XXX
    schema1_std = []     # STD for schema 1XXX
    schema0_avg = []     # Average fitness for schema 0XXX
    schema0_std = []     # STD for schema 0XXX
    
    for gen in range(generations):
        population = np.random.permutation(population)  # Shuffle 
        new_population = population.copy()
        fitness_values = []
        
        error_count = 0
        correct_count = 0
        
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i]['individual'], population[i+1]['individual']
            child1, child2 = crossover(parent1, parent2)
            
   
            candidates = [parent1, parent2, child1, child2]
            candidate_fitness = np.array([fitness_func(s) for s in candidates])
            
            sorted_indices = np.argsort(candidate_fitness)[::-1]  # sort descending
            best1, best2 = candidates[sorted_indices[0]], candidates[sorted_indices[1]]
            best1_fitness, best2_fitness = candidate_fitness[sorted_indices[0]], candidate_fitness[sorted_indices[1]]
            
            new_population[i] = (best1, best1_fitness)
            new_population[i + 1] = (best2, best2_fitness)
            fitness_values.append(best1_fitness)
            fitness_values.append(best2_fitness)
            
            # Count selection decisions 
            for pos in range(40):  
                if parent1[pos] != parent2[pos]:
                    # If winners have bit-0 -> selection error
                    if best1[pos] == '0' and best2[pos] == '0':
                        error_count += 1
                    # If the winners have a bit-1 -> correct selection decisinon
                    elif best1[pos] == '1' and best2[pos] == '1':
                        correct_count += 1
        
        #  proportion calculation (1s in pop)
        total_bits_1 = sum(ind['individual'].count('1') for ind in population)
        proportion = total_bits_1 / (len(population) * 40)
        prop_t.append(proportion)
        errors_t.append(error_count)
        correct_t.append(correct_count)
        
        # schemata metrics on the evolved generation (new_population)
        # Schemata: "1********…" versus "0********…"
        schema1_inds = [ind for ind in new_population if ind['individual'][0] == '1']
        schema0_inds = [ind for ind in new_population if ind['individual'][0] == '0']
        schema1_count.append(len(schema1_inds))
        schema0_count.append(len(schema0_inds))
        
        if len(schema1_inds) > 0:
            s1_fit = np.array([ind['fitness'] for ind in schema1_inds])
            schema1_avg.append(np.mean(s1_fit))
            schema1_std.append(np.std(s1_fit))
        else:
            schema1_avg.append(0)
            schema1_std.append(0)
            
        if len(schema0_inds) > 0:
            s0_fit = np.array([ind['fitness'] for ind in schema0_inds])
            schema0_avg.append(np.mean(s0_fit))
            schema0_std.append(np.std(s0_fit))
        else:
            schema0_avg.append(0)
            schema0_std.append(0)
        
        # Print generation details (including new schemata metrics)
        print(f"Generation {gen}: Proportion of 1s = {proportion:.4f}, Errors = {error_count}, Correct = {correct_count}")
        print(f"   Schema 1 count = {schema1_count[-1]}, Avg fitness = {schema1_avg[-1]:.4f}, Std dev = {schema1_std[-1]:.4f}")
        print(f"   Schema 0 count = {schema0_count[-1]}, Avg fitness = {schema0_avg[-1]:.4f}, Std dev = {schema0_std[-1]:.4f}")
        
        population = new_population
        
        if proportion == 1.0:
            break
    
    return population, prop_t, errors_t, correct_t, schema1_count, schema0_count, schema1_avg, schema1_std, schema0_avg, schema0_std

if __name__ == "__main__":
    # Ensure "plots" directory exists
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    config = {
        'pop_size': 200,
        'generations': 100,
        'fitness_func': counting_ones,
        'crossover': uniform_crossover
    }
    
    start_time = time.time()
    population = init_pop_generation(config['pop_size'], fitness_func=config['fitness_func'])
    results = evolve_population(
        population, config['generations'], fitness_func=config['fitness_func'], crossover=config['crossover']
    )
    
    # Unpack returned metrics
    final_population, prop_t, errors_t, correct_t, schema1_count, schema0_count, schema1_avg, schema1_std, schema0_avg, schema0_std = results
    elapsed_time = time.time() - start_time
    
    # Proportion of 1s
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(prop_t))), y=prop_t, mode='lines', name='Proportion of 1-bits'))
    fig.update_layout(title='Proportion of 1-bits over Generations',
                     xaxis_title='Generation',
                     yaxis_title='Proportion of 1s',
                     template='plotly_white')
    fig.write_image(os.path.join(plot_dir, "ga_proportion_plot.png"))

    # Selection Decisions 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(errors_t))), y=errors_t, mode='lines', name='Selection Errors Err(t)'))
    fig.add_trace(go.Scatter(x=list(range(len(correct_t))), y=correct_t, mode='lines', name='Correct Selections Correct(t)'))
    fig.update_layout(title='Selection Decisions over Generations',
                      xaxis_title='Generation',
                      yaxis_title='Count',
                      template='plotly_white')
    fig.write_image(os.path.join(plot_dir, "ga_selection_decisions_plot.png"))
    
    # Average Fitness for each schema over generations
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(x=list(range(len(schema1_avg))), y=schema1_avg, mode='lines', name='Schema 1 Avg Fitness'))
    fig_avg.add_trace(go.Scatter(x=list(range(len(schema0_avg))), y=schema0_avg, mode='lines', name='Schema 0 Avg Fitness'))
    fig_avg.update_layout(title='Schema Average Fitness over Generations',
                      xaxis_title='Generation',
                      yaxis_title='Average Fitness',
                      template='plotly_white')
    fig_avg.write_image(os.path.join(plot_dir, "ga_schemata_avg_fitness.png"))
    
    # Standard Deviation for each schema over generations
    fig_std = go.Figure()
    fig_std.add_trace(go.Scatter(x=list(range(len(schema1_std))), y=schema1_std, mode='lines', name='Schema 1 Std Dev'))
    fig_std.add_trace(go.Scatter(x=list(range(len(schema0_std))), y=schema0_std, mode='lines', name='Schema 0 Std Dev'))
    fig_std.update_layout(title='Schema Standard Deviation over Generations',
                      xaxis_title='Generation',
                      yaxis_title='Standard Deviation',
                      template='plotly_white')
    fig_std.write_image(os.path.join(plot_dir, "ga_schemata_std_dev.png"))
    
    print(f"Total runtime: {elapsed_time:.4f} seconds")
    print(f"Plots saved in '{plot_dir}/' directory.")

