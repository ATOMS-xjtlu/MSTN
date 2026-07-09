import matplotlib.pyplot as plt
import numpy as np


def read_sum_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                cols = line.split()
                if len(cols) >= 8:
                    residue = cols[0]
                    probabilities = list(map(float, cols[1:8]))
                    data.append([residue] + probabilities)
    return data


def plot_summary(
    input_file,
    output_file,
    bar_colors,
    custom_labels,
    title='Dictionary of Secondary Structure of Protein ',
):
    data = read_sum_file(input_file)

    target_data = data[6:15]
    residues = [row[0] for row in target_data]
    beta_vals = [row[1] + row[2] for row in target_data]
    helix_vals = [row[3] + row[4] + row[5] for row in target_data]
    turn_vals = [row[6] + row[7] for row in target_data]

    positions = np.arange(len(residues))

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        positions,
        beta_vals,
        color=bar_colors['beta'],
        label='BETA',
        edgecolor='gray',
        linewidth=0.5,
    )
    ax.bar(
        positions,
        helix_vals,
        bottom=beta_vals,
        color=bar_colors['helix'],
        label='HELIX',
        edgecolor='gray',
        linewidth=0.5,
    )
    ax.bar(
        positions,
        turn_vals,
        bottom=[beta_vals[i] + helix_vals[i] for i in range(len(beta_vals))],
        color=bar_colors['turn'],
        label='TURN',
        edgecolor='gray',
        linewidth=0.5,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(custom_labels, rotation=45, ha='right', fontsize=20)
    ax.set_ylabel('Probability', fontsize=20)
    ax.set_xlabel('Residue', fontsize=20)
    ax.set_title(title, fontsize=20)
    # ax.legend(loc='upper right', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.19)
    ax.tick_params(axis='y', labelsize=20)

    for i in range(len(residues)):
        base_values = [0, beta_vals[i], beta_vals[i] + helix_vals[i]]
        structure_values = [beta_vals[i], helix_vals[i], turn_vals[i]]

        for bottom, value in zip(base_values, structure_values):
            if value > 0.05:
                ax.text(
                    i,
                    bottom + value / 2,
                    f'{value:.2f}',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=20,
                    fontweight='bold',
                )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_summary(
        input_file='Pre-active dssp/replica3/1us/dssp50-65.gnu.sum',
        output_file='Pre-active dssp/replica3/1us/column_dssp50-65.png',
        bar_colors={
            'beta': '#faa26f',
            'helix': '#b0d6a9',
            'turn': '#3c9bc9',
        },
        custom_labels=['PRO', 'HIE', 'THR', 'HIE', 'LEU', 'VAL', 'HIE', 'GLN', 'ALA'],
    )