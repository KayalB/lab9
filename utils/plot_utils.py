import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plot_rewards(rewards, title, save_path):
    plt.figure()
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_path_visualization(env, path, title, save_path):
    """
    Visualize the agent's movement path through the maze.
    
    Args:
        env: MazeEnvironment instance
        path: List of (row, col) tuples representing the agent's path
        title: Title for the plot
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a color map for the maze
    # 0 = white (free space), 1 = black (wall), 2 = green (goal)
    maze_display = np.copy(env.grid)
    
    # Create custom colormap
    cmap_colors = ['white', 'black', 'green']
    
    # Draw the base maze
    for i in range(env.N):
        for j in range(env.M):
            if env.grid[i, j] == 1:  # Wall
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='black'))
            elif (i, j) == env.goal:  # Goal
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightgreen', edgecolor='green', linewidth=2))
            elif (i, j) == env.start:  # Start
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightblue', edgecolor='blue', linewidth=2))
            else:  # Free space
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='lightgray'))
    
    # Draw the path
    if len(path) > 1:
        path_array = np.array(path)
        # Plot the path line
        ax.plot(path_array[:, 1] + 0.5, path_array[:, 0] + 0.5, 
                'r-', linewidth=2, alpha=0.6, label='Path')
        
        # Add arrows to show direction
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            # Only draw arrow if it's a valid move (not a repeated position)
            if (y1, x1) != (y2, x2):
                ax.annotate('', xy=(x2 + 0.5, y2 + 0.5), xytext=(x1 + 0.5, y1 + 0.5),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7))
        
        # Mark start and end positions with circles
        ax.plot(path[0][1] + 0.5, path[0][0] + 0.5, 'bo', markersize=15, label='Start')
        ax.plot(path[-1][1] + 0.5, path[-1][0] + 0.5, 'go', markersize=15, label='End')
    
    # Set axis properties
    ax.set_xlim(0, env.M)
    ax.set_ylim(0, env.N)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Add grid
    ax.set_xticks(np.arange(0, env.M + 1, 1))
    ax.set_yticks(np.arange(0, env.N + 1, 1))
    ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='blue', label='Start'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='green', label='Goal'),
        mpatches.Patch(facecolor='black', label='Wall'),
        mpatches.Patch(facecolor='red', label='Path', alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Path visualization saved to {save_path}")
    print(f"Path length: {len(path)} steps")

def plot_path_heatmap(env, all_paths, title, save_path):
    """
    Visualize the agent's movement patterns using a heatmap showing visit frequency.
    
    Args:
        env: MazeEnvironment instance
        all_paths: List of paths, where each path is a list of (row, col) tuples
        title: Title for the plot
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap matrix to count visits
    visit_counts = np.zeros((env.N, env.M))
    
    # Count visits from all paths
    total_steps = 0
    for path in all_paths:
        for pos in path:
            if len(pos) == 2:  # Ensure valid position
                visit_counts[pos[0], pos[1]] += 1
                total_steps += 1
    
    # Normalize by number of paths for better interpretation
    visit_frequency = visit_counts / len(all_paths) if all_paths else visit_counts
    
    # Draw the base maze with heatmap
    for i in range(env.N):
        for j in range(env.M):
            if env.grid[i, j] == 1:  # Wall
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='black', edgecolor='gray'))
                ax.text(j + 0.5, i + 0.5, 'WALL', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
            else:
                # Use heatmap coloring for free spaces
                intensity = visit_frequency[i, j]
                if intensity > 0:
                    # Use a color gradient from white (low) to red (high)
                    color = plt.cm.Reds(min(intensity / visit_frequency.max(), 1.0))
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='gray', linewidth=0.5))
                    # Add visit count text
                    count = int(visit_counts[i, j])
                    ax.text(j + 0.5, i + 0.5, str(count), ha='center', va='center',
                           color='black' if intensity < visit_frequency.max() * 0.5 else 'white',
                           fontsize=10, fontweight='bold')
                else:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='gray', linewidth=0.5))
    
    # Highlight start and goal positions with borders and markers
    start_y, start_x = env.start
    goal_y, goal_x = env.goal
    
    # Draw thick borders for start and goal
    ax.add_patch(plt.Rectangle((start_x, start_y), 1, 1, fill=False, 
                               edgecolor='blue', linewidth=4))
    ax.text(start_x + 0.5, start_y + 0.1, 'START', ha='center', va='top',
           color='blue', fontsize=10, fontweight='bold', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    ax.add_patch(plt.Rectangle((goal_x, goal_y), 1, 1, fill=False,
                               edgecolor='green', linewidth=4))
    ax.text(goal_x + 0.5, goal_y + 0.9, 'GOAL', ha='center', va='bottom',
           color='green', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(0, env.M)
    ax.set_ylim(0, env.N)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Add grid
    ax.set_xticks(np.arange(0, env.M + 1, 1))
    ax.set_yticks(np.arange(0, env.N + 1, 1))
    ax.grid(True, color='gray', linewidth=0.5, alpha=0.3)
    
    # Labels and title
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    
    # Create a separate axis for colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = Normalize(vmin=0, vmax=visit_frequency.max())
    cbar = ColorbarBase(cbar_ax, cmap=plt.cm.Reds, norm=norm, orientation='vertical')
    cbar.set_label('Average Visits per Episode', fontsize=11)
    
    # Add statistics text
    stats_text = f'Total Paths: {len(all_paths)}\n'
    stats_text += f'Total Steps: {total_steps}\n'
    stats_text += f'Avg Steps/Path: {total_steps/len(all_paths):.1f}\n'
    stats_text += f'Max Visits: {int(visit_counts.max())}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap visualization saved to {save_path}")
    print(f"Total paths analyzed: {len(all_paths)}")
    print(f"Total steps: {total_steps}")
    print(f"Average steps per path: {total_steps/len(all_paths):.1f}")
