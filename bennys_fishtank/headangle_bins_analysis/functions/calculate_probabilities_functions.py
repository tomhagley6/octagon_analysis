from collections import defaultdict
import numpy as np

def calculate_high_wall_probabilities(bin_assignments_lists, high_wall_chosen_lists, debug=False):
    
    # Creating dictionaries to store counts
    bin_total_counts = defaultdict(int)
    bin_true_counts = defaultdict(int)

    #trial_data = [(bin_assignments_list, high_wall_chosen_list)]

    for bin_assignments, high_wall_chosen in zip(bin_assignments_lists, high_wall_chosen_lists):
        
        for bin_index, is_high_wall_chosen in zip(bin_assignments, high_wall_chosen):

            if np.isnan(is_high_wall_chosen):  # Ignore NaN values
                continue

            bin_total_counts[bin_index] += 1  # Total count per bin

            if is_high_wall_chosen:  # Count if high wall was chosen
                bin_true_counts[bin_index] += 1

    # Calculate probabilities
    probabilities_dict = {
        bin_index: bin_true_counts[bin_index] / bin_total_counts[bin_index]
        for bin_index in sorted(bin_total_counts)
    }

    # Convert to a list ordered by bin index
    probabilities_list = [probabilities_dict[bin_index] for bin_index in sorted(bin_total_counts)]

    # Debugging output
    if debug:
        print(probabilities_list)
        for bin_index, probability in sorted(probabilities_dict.items()):
            print(f"Bin {bin_index}: {probability:.2f} (True/Total = {bin_true_counts[bin_index]}/{bin_total_counts[bin_index]})")

    return probabilities_dict, probabilities_list





def calculate_p_high(bin_assignments_lists, high_wall_chosen_lists):
    
    true_counts = 0
    total_counts = 0
    
    #trial_data = [(bin_assignments_list, high_wall_chosen_list)]
    
    for bin_assignments, high_wall_chosen in zip(bin_assignments_lists, high_wall_chosen_lists):
        
        for is_high_wall_chosen in high_wall_chosen:
            
            if np.isnan(is_high_wall_chosen):  
                continue         
                
            total_counts += 1  #increment total trial count for the bin
            
            if is_high_wall_chosen:          #increment True count if high wall is chosen
                true_counts += 1
                
    overall_probability = true_counts / total_counts

    return overall_probability




