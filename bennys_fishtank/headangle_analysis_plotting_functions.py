import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import math
import plotting.plot_octagon as plot_octagon



def plot_ratios_under_alcoves(ax, ratios_list):
    ''' Function to plot the values from ratios_list under each alcove '''
    
    # Get the alcove center points
    alcove_centers = plot_octagon.return_alcove_centre_points()
    print(alcove_centers.shape)

    # Iterate over each alcove and place the value from ratios_list
    for i in range(len(ratios_list)):
        # Get the center of the current alcove
        alcove_center_x, alcove_center_y = alcove_centers[:,i]
        
        # Get the ratio corresponding to this alcove
        ratio = ratios_list[i]
        
        # Add text under the alcove center
        ax.text(alcove_center_x, alcove_center_y - 0.15, f'{ratio:.2f}', 
                ha='center', va='center', fontsize=8, color='black')
    
    return ax





def plot_colored_octagon(ax, bin_ranges, ratios_list, radius=18):
    ''' Function to color octagon segments based on ratios '''
    
    # Normalize the ratios to [0, 1] for colormap
    norm = plt.Normalize(vmin=min(ratios_list), vmax=max(ratios_list))
    colormap = cm.RdBu  # Use Red-Blue colormap

    # Reverse the ratios list to match clockwise order
    ratios_list_reversed = ratios_list[::-1]

    # Define the angular shift (to rotate by 3 segments counterclockwise)
    angular_shift = 3 * np.pi / 4  # This corresponds to 3 octagon segments
    

    # Get the alcove center points (used later for positioning text)
    alcove_centers = plot_octagon.return_alcove_centre_points()
    
    # Iterate over each segment and plot it
    for idx, (ratio, (start_angle, end_angle)) in enumerate(zip(ratios_list_reversed, bin_ranges)):
        #print(f"Segment {idx + 1}: Start angle: {start_angle:.2f}, End angle: {end_angle:.2f}, Ratio: {ratio:.2f}")
        # Apply the angular shift to both start and end angles
        start_angle += angular_shift
        end_angle += angular_shift

        # Normalize angles to keep them in the [0, 2*pi] range
        start_angle = start_angle % (2 * np.pi)
        end_angle = end_angle % (2 * np.pi)
        
        # Generate points for the straight segment (cone shape)
        # Define the boundary points on the octagon at the specified angles
        x1 = radius * np.cos(start_angle)
        y1 = radius * np.sin(start_angle)
        x2 = radius * np.cos(end_angle)
        y2 = radius * np.sin(end_angle)
        
        # Create the polygon with straight lines: from center (0, 0) to each boundary point
        polygon_x = [0, x1, x2, 0]
        polygon_y = [0, y1, y2, 0]
        
        # Get the color for the current ratio
        color = colormap(norm(ratio))  # Map ratio to color
        #print(f"Segment {idx + 1}: Ratio {ratio:.2f}, Assigned Color: {color}")
        
        # Plot the filled segment
        ax.fill(polygon_x, polygon_y, color=color, alpha=0.8, edgecolor='black')
    
    return ax