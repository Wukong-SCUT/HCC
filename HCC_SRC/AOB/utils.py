import matplotlib.pyplot as plt
import numpy as np

'''
This library provides utility functions for AOB:
1. calculate_DA: Computes decomposition accuracy.
2. remove_overlapping_groups: Removes overlapping groups.
3. make_monotonic_decreasing: Makes an array monotonically decreasing.
4. combine: Maps subspaces back to the full space in CC.
5. evaluation_record: Records evaluation data.
6. plot_evaluation_curve: Plots evaluation curve.
7. plot_evaluation_curve_best_so_far: Plots best-so-far evaluation curve.
'''

def calculate_DA(true_subcomps, predicted_subcomps):
    """
    Calculate Decomposition Accuracy (DA).

    Args:
        true_subcomps (list of sets): True subcomponents, each subcomponent is a set.
        predicted_subcomps (list of sets): Predicted subcomponents, each subcomponent is a set.

    Returns:
        float: DA value.
    """
    numerator = 0
    denominator = 0

    # Loop through each true subcomponent
    for true_comp in true_subcomps:
        true_set = set(true_comp)
        intersections = [len(true_set & set(pred_comp)) for pred_comp in predicted_subcomps]
        numerator += max(intersections)
        denominator += len(true_comp)

    # Compute DA
    da = numerator / denominator if denominator > 0 else 0
    return da

def remove_overlapping_groups(grouping_result):
    """
    Remove overlapping groups and return the unique groups, duplicates, and overlap information.

    Args:
        grouping_result (list of lists): List of groups, each group is a list of items.

    Returns:
        tuple: A tuple containing:
            - unique_groups (list of lists): Unique groups after removing overlaps.
            - duplicates (list): List of duplicate items across all groups.
            - overlap_groups (list of lists): List of overlap between adjacent groups.
    """
    seen = set()
    unique_groups = []
    duplicates = []
    overlap_groups = []
    grouping_result_ = grouping_result.copy()  # Avoid modifying the original array

    # Identify duplicates in all groups
    for group in grouping_result:
        for item in group:
            if item in seen:
                duplicates.append(item)
            else:
                seen.add(item)

    # Process groups and remove duplicates, calculate overlaps
    for i in range(len(grouping_result) - 1):
        group1 = grouping_result[i]
        group2 = grouping_result[i + 1]
        group1_ = grouping_result_[i]
        group2_ = grouping_result_[i + 1]

        overlap = set(group1) & set(group2)
        overlap_groups.append(list(overlap))

        unique_group1 = [item for item in group1_ if item not in overlap]
        unique_group2 = [item for item in group2_ if item not in overlap]

        unique_groups.append(unique_group1)
        grouping_result_[i + 1] = unique_group2

    unique_groups.append(list(set(grouping_result[-1]) - set(overlap_groups[-1])))
    duplicates = list(set(duplicates))
    return unique_groups, duplicates, overlap_groups

def make_monotonic_decreasing(arr):
    """
    Modify the array to make it monotonically decreasing.

    Args:
        arr (list or np.ndarray): The input array.

    Returns:
        np.ndarray: The modified array.
    """
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:
            arr[i + 1] = arr[i]
    return arr

def combine(small_vec, background_vec, location):
    """
    Combine a small vector into a background vector at a specified location.

    Args:
        small_vec (np.ndarray): The small vector to be combined.
        background_vec (np.ndarray): The background vector.
        location (int or None): The location in the background vector where the small vector will be placed.

    Returns:
        np.ndarray: The combined vector.
    """
    if location is None:
        return small_vec
    else:
        combination = np.tile(background_vec, (len(small_vec), 1))
        combination[:, location] = small_vec
        return combination

def evaluation_record(data, output_path, record_FEs_list):
    """
    Record evaluation values of the algorithm (including specific evaluation points and final value) and execution time.

    Args:
        data (dict): Dictionary containing algorithm results and execution time.
        output_path (str): Path to save the output file.
        record_FEs_list (list): List of specific evaluation points to record.
    """
    record_FEs_list = [int(x) for x in record_FEs_list]

    algorithm_avg_fitness = {}

    for algorithm, runs in data.items():
        if "_time" in algorithm:
            continue

        runs = [make_monotonic_decreasing(run.copy()) for run in runs]
        max_length = max(len(run) for run in runs)
        avg_fitness = []

        for i in range(max_length):
            values_at_i = [run[i] for run in runs if i < len(run)]
            if values_at_i:
                avg_fitness.append(sum(values_at_i) / len(values_at_i))
            else:
                avg_fitness.append(None)

        algorithm_avg_fitness[algorithm] = avg_fitness

    output_file_path = f"{output_path}evaluation_record.txt"
    with open(output_file_path, 'w') as f:
        f.write(f"{'Algorithm':<20}{'Record Point':<25}{'Fitness Value':<40}{'Scientific Notation':<25}\n")
        f.write("-" * 120 + "\n")

        for algorithm, avg_fitness in algorithm_avg_fitness.items():
            f.write(f"Algorithm: {algorithm}\n")

            time_key = f"{algorithm}_time"
            if time_key in data:
                run_time = data[time_key][0]
            else:
                run_time = None

            for record_FEs in record_FEs_list:
                if record_FEs < len(avg_fitness):
                    fitness_value = avg_fitness[record_FEs]
                    if fitness_value is not None:
                        f.write(f"{'':<20}{record_FEs:<25.3e}{fitness_value:<40.6f}{fitness_value:<25.6e}\n")
                else:
                    print(f"Warning: Record point {record_FEs} exceeds the available evaluations for {algorithm}.")

            final_value = avg_fitness[-1] if avg_fitness[-1] is not None else "N/A"
            f.write(f"{'':<20}{f'Fin:{len(avg_fitness):.3e}':<25}{final_value:<40.6f}{final_value:<25.6e}\n")

            if run_time is not None:
                f.write(f"{'':<20}{'Run Time:':<25}{run_time:<40.6f}{run_time:<25.6e}\n")

            f.write("-" * 120 + "\n")

    print(f"Evaluation records have been saved to '{output_file_path}'.")

def plot_evaluation_curve(data, output_path, font_size, log_scale=False, show_variance=False):
    """
    Plot the evaluation curve of different algorithms, with an optional variance band.

    Args:
        data (dict): Dictionary containing the evaluation values of algorithms, in the format:
                     {"Algorithm1": [[list1], [list2], ...], "Algorithm2": [[list1], [list2], ...], ...}
        log_scale (bool): Whether to use logarithmic scale for the y-axis.
        show_variance (bool): Whether to display the variance band.
    """
    plt.figure(figsize=(6, 6))

    for algorithm, runs in data.items():
        if "_time" in algorithm:
            continue

        max_length = max(len(run) for run in runs)
        avg_fitness = []
        std_fitness = []

        for i in range(max_length):
            values_at_i = [run[i] for run in runs if i < len(run)]
            if values_at_i:
                avg_fitness.append(np.mean(values_at_i))
                std_fitness.append(np.std(values_at_i))
            else:
                avg_fitness.append(None)
                std_fitness.append(None)

        x = range(len(avg_fitness))
        y = [v for v in avg_fitness if v is not None]
        x = [i for i, v in enumerate(avg_fitness) if v is not None]
        std = [s for s in std_fitness if s is not None]

        line, = plt.plot(x, y, label=algorithm)

        if show_variance and std:
            plt.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                             color=line.get_color(), alpha=0.2)

    if log_scale:
        plt.yscale("log")

    plt.rcParams.update({'font.size': font_size})
    plt.xlabel("FEs", fontsize=font_size)
    plt.ylabel("Objective Value (log10)", fontsize=font_size)
    plt.title("Best-so-Far Evaluation Curves for Different Algorithms", fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)

    filename = "evaluation_curves.png"
    plt.savefig(f"{output_path}{filename}", bbox_inches='tight')
    print(f"Plot saved to '{output_path}{filename}'.")
    plt.close()

def plot_evaluation_curve_best_so_far(data, output_path, font_size, log_scale=False, show_variance=False):
    """
    Plot the best-so-far evaluation curve of different algorithms, with an optional variance band.

    Args:
        data (dict): Dictionary containing the evaluation values of algorithms, in the format:
                     {"Algorithm1": [[list1], [list2], ...], "Algorithm2": [[list1], [list2], ...], ...}
        log_scale (bool): Whether to use logarithmic scale for the y-axis.
        show_variance (bool): Whether to display the variance band.
    """
    plt.figure(figsize=(6, 6))

    for algorithm, runs in data.items():
        if "_time" in algorithm:
            continue

        runs = [make_monotonic_decreasing(run.copy()) for run in runs]

        max_length = max(len(run) for run in runs)
        avg_fitness = []
        std_fitness = []

        for i in range(max_length):
            values_at_i = [run[i] for run in runs if i < len(run)]
            if values_at_i:
                avg_fitness.append(np.mean(values_at_i))
                std_fitness.append(np.std(values_at_i))
            else:
                avg_fitness.append(None)
                std_fitness.append(None)

        x = range(len(avg_fitness))
        y = [v for v in avg_fitness if v is not None]
        x = [i for i, v in enumerate(avg_fitness) if v is not None]
        std = [s for s in std_fitness if s is not None]

        line, = plt.plot(x, y, label=algorithm)

        if show_variance and std:
            plt.fill_between(x, np.array(y) - np.array(std), np.array(y) + np.array(std),
                             color=line.get_color(), alpha=0.2)

    if log_scale:
        plt.yscale("log")

    plt.rcParams.update({'font.size': font_size})
    plt.xlabel("FEs", fontsize=font_size)
    plt.ylabel("Objective Value (log10)", fontsize=font_size)
    plt.title("Best-so-Far Evaluation Curves for Different Algorithms", fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)

    filename = "evaluation_curves_best_so_far.png"
    plt.savefig(f"{output_path}{filename}", bbox_inches='tight')
    print(f"Plot saved to '{output_path}{filename}'.")
    plt.close()





