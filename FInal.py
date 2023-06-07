import random
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from deap.benchmarks.tools import diversity, convergence, hypervolume
import matplotlib.pyplot as plt
import pandas as pd
import json

# Load the Sonar dataset from .data file
sonar_data = np.genfromtxt(
    '/Users/macbook/Desktop/Final_sonar/sonar.data', delimiter=',', dtype=np.float32)
X = sonar_data[:, :-1]
y = sonar_data[:, -1]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the problem
creator.create("FitnessMin", base.Fitness,
               weights=(-1.0, -1.0))  # Minimization problem
creator.create("Individual", list, fitness=creator.FitnessMin)


def convert_to_json_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(
        "Object of type {} is not JSON serializable".format(type(obj)))


def evaluate_2(individual):
    selected_features = [idx for idx,
                         feature in enumerate(individual) if feature]

    if len(selected_features) == 0:
        return float('inf'),  # Return infinity if no features are selected
    X_train_selected = X_train_scaled[:, selected_features]
    X_test_selected = X_test_scaled[:, selected_features]

    # Fit KNN classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_selected, y_train)

    # Predict labels for test data
    y_pred = clf.predict(X_test_selected)

    # Calculate classification error (1 - accuracy)
    error = 1.0 - accuracy_score(y_test, y_pred)

    return error


def evaluate(individual):
    # Define the objectives to be optimized
    [min_obj1, max_obj1] = [0, 1]
    [min_obj2, max_obj2] = [1, 60]

    obj1 = evaluate_2(individual)
    selected_features = [idx for idx,
                         feature in enumerate(individual) if feature]
    obj2 = len(selected_features)
    norm_obj1 = (obj1-min_obj1)/(max_obj1-min_obj1)
    norm_obj2 = (obj2-min_obj2)/(max_obj2-min_obj2)
    return obj1, obj2


toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.choice, [0, 1])
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, n=len(X[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Define the genetic operators
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
toolbox.register("select", tools.selNSGA2)

toolbox.register("evaluate", evaluate)

# Initialize results variables
results = []
gen_ls = []
Fitness_err_ls = []
Fitness_feat_ls = []
Saved_para = []
hyperolume_ls = []
selected_features_ls = []


def main():
    random.seed(42)

    # Create the population
    pop = toolbox.population(n=100)
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop)
    for ind in pop:
        if not ind.fitness.valid:
            fitness = toolbox.evaluate(ind)
            ind.fitness.values = fitness
    Intial_front = tools.sortNondominated(pop, len(pop))
    file_name = "pareto_front_intial_sonar.txt"  # Choose a suitable file name
    with open(file_name, 'w') as file:
        for ind in Intial_front[0]:
            file.write(f"{ind.fitness.values[0]} {ind.fitness.values[1]}\n")
    final_front = None



    # Initialize lists to store best individual and its properties for each generation
    best_individuals = []
    best_errors = []
    best_num_features = []
    lengths_of_features = []
    ngen = 15
    mu = 100
    max_nfc = 100 * 10  # Termination condition: Max NFC

    # Run the evolution
    for gen in range(1, ngen + 1):
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        nfc = 0  # Number of function calls
        while nfc < max_nfc:
            offspring = [toolbox.clone(ind) for ind in pop]
            for ind in offspring:
                if random.random() < 0.05:
                    toolbox.mutate(ind)
                del ind.fitness.values

            nfc += len(offspring)

            invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_inds)
            for ind, fit in zip(invalid_inds, fitnesses):
                ind.fitness.values = fit

            pop[:] = toolbox.select(pop + offspring, mu)
        for ind in pop:
            if not ind.fitness.valid:
                fitness = toolbox.evaluate(ind)
                ind.fitness.values = fitness
        pareto_front.update(pop)

        # Get the best individual from the current population
        best_individual = tools.selBest(pop, k=1)[0]
        # Evaluate the best individual
        individual_index = pop.index(best_individual)
        best_error, best_num = evaluate(best_individual)
        selected_features_temp = [idx for idx,
                                  feature in enumerate(best_individual) if feature]
        print(f'generation\tClassification_error\t number_of_feattures\tIndexofindividual')
        print(
            f'{gen}\t{best_individual.fitness.values[0]}\t{best_individual.fitness.values[1]}\t{individual_index}')
        gen_ls.append(gen)
        Fitness_feat_ls.append(best_individual.fitness.values[1])
        Fitness_err_ls.append(best_individual.fitness.values[0])
        if gen == ngen:

            final_front = tools.sortNondominated(pop, len(pop))
            file_name = "pareto_front_final_sonar.txt"  # Choose a suitable file name
            with open(file_name, 'w') as file:
                for ind in final_front[0]:
                    file.write(f"{ind.fitness.values[0]} {ind.fitness.values[1]}\n")
    
        # Append the best individual and its properties to the lists
        best_individuals.append(best_individual)
        best_errors.append(best_error)
        best_num_features.append(best_num)
        lengths_of_features.append(len(best_individual))

        selected_features = [idx for idx,
                             feature in enumerate(best_individual) if feature]

        X_train_selected = X_train_scaled[:, selected_features]
        X_test_selected = X_test_scaled[:, selected_features]

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_selected, y_train)

        y_train_pred = clf.predict(X_train_selected)
        train_error = 1.0 - accuracy_score(y_train, y_train_pred)

        y_test_pred = clf.predict(X_test_selected)
        test_error = 1.0 - accuracy_score(y_test, y_test_pred)

        # Store the results
        result = {
            'Min_Classification_Error_Train': train_error,
            'Min_Classification_Error_Test': test_error,
            'Num_Selected_Features': len(selected_features),
            'Selected_Features': selected_features,
            'population': pop,
            'index': individual_index
        }
        hv = hypervolume(tools.sortNondominated(pop, len(pop))[0])
        hyperolume_ls.append(hv)
        selected_features_ls.append(selected_features_temp)
        print(hv)

        # ******************************* array becoming null after loop exits ********************
        results.append(result)

    return results, Fitness_err_ls, Fitness_feat_ls, gen_ls, Intial_front, final_front


if __name__ == "__main__":
    results, fitness_err_ls, fitnes_feat_ls, Gen_ls, firt_p, second_p = main()

    generation = range(1, 16)  # Assuming there are 15 generations

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    data = zip(gen_ls, fitnes_feat_ls, fitness_err_ls)
    filename = 'GEN_FITNESS_SONAR.csv'
    Minimum_train_error_ls = []
    Minimum_test_error_ls = []
    number_of_features_ls = []
    index_ls = []

    Gen_number = [idx+1 for idx in range(0, len(results))]
    for index in range(0, len(results)):
        Minimum_train_error_ls.append(
            results[index]['Min_Classification_Error_Train'])
        Minimum_test_error_ls.append(
            results[index]['Min_Classification_Error_Test'])
        number_of_features_ls.append(results[index]['Num_Selected_Features'])
        index_ls.append(results[index]['index'])

    results_gen = pd.DataFrame(columns=['generations', 'Number of Features',
                                        'min classification error on train', 'min classification on test', 'index of selected feature', 'hypervolume', 'selected features'])


# Dictionary containing the array data
    data = {
        'generations': gen_ls,  # Example array for 'generations'
        # Example array for 'Number of Features'
        'Number of Features': number_of_features_ls,
        # Example array for 'min classification error on train'
        'min classification error on train': Minimum_train_error_ls,
        # Example array for 'min classification on test'
        'min classification on test': Minimum_test_error_ls,
        'index of selected feature': index_ls,
        'hypervolume': hyperolume_ls,
        'selected features': selected_features_ls
    }

# Convert dictionary to DataFrame and append to existing DataFrame
    new_data = pd.DataFrame(data)
    results_gen = pd.concat([results_gen, new_data], ignore_index=True)
    print('result_gen data intialized')

# Save the DataFrame as a CSV file
    results_gen.to_csv('results_gen_min_index.csv', index=False)
    print('Data saved successfully in', filename)
# Plotting fitness error
    ax1.plot(generation, fitness_err_ls)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Error')
    ax1.set_title('Generation vs. Fitness Error')

# Plotting fitness feature
    ax2.plot(generation, fitnes_feat_ls)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Feature')
    ax2.set_title('Generation vs. Fitness Feature')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    print('ploting pareto')
    population = results[0]['population']
    population_2 = results[14]['population']

    print("generating pareto pront values")
    pareto_front = np.array([ind.fitness.values for ind in firt_p[0]])
    pareto_front_2 = np.array([ind.fitness.values for ind in second_p[0]])
    # Plot the initial and final Pareto fronts
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], color='b', marker='o', linestyle=':')
    plt.title("Initial Pareto Front")
    plt.xlabel("Classification error")
    plt.ylabel("Number of features")

    plt.subplot(1, 2, 2)
    plt.plot(pareto_front_2[:, 0], pareto_front_2[:, 1], color='r', marker='o', linestyle=':')
    plt.title("Final Pareto Front")
    plt.xlabel("classification error")
    plt.ylabel("Number of features")

    plt.tight_layout()
    plt.show()
    with open('results_sonar.txt', 'w') as file:
        for result in results:
            file.write(str(result) + '\n')


# Calculate average HV over 15 runs
# hv_sum = 0.0
#  for result in results:
#     final_pareto_front = result['Final_Pareto_Front']
#    hv = tools.hypervolume(final_pareto_front)
#   hv_sum += hv
# average_hv = hv_sum / len(results)
# print("Average HV over 15 runs:", average_hv)

print("End")