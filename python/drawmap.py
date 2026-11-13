from PIL import Image
# import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def color_matrix_to_image(color_matrix):
    """
    Convert a color-code matrix into an image displayable by pyplot.

    Parameters:
    color_matrix: 2D list or numpy array of color-code strings
                  e.g. '#FF0000', 'red', 'rgb(255,0,0)', 'r'
    """
    rows = len(color_matrix)
    cols = len(color_matrix[0]) if rows > 0 else 0

    rgb_array = np.zeros((rows, cols, 3), dtype=float)

    # def color_to_rgb(color_str):
    #     color_str = str(color_str).strip().lower()
    #
    #     color_dict = {
    #         'r': (1, 0, 0), 'red': (1, 0, 0),
    #         'g': (0, 1, 0), 'green': (0, 1, 0),
    #         'b': (0, 0, 1), 'blue': (0, 0, 1),
    #         'y': (1, 1, 0), 'yellow': (1, 1, 0),
    #         'm': (1, 0, 1), 'magenta': (1, 0, 1),
    #         'c': (0, 1, 1), 'cyan': (0, 1, 1),
    #         'w': (1, 1, 1), 'white': (1, 1, 1),
    #         'k': (0, 0, 0), 'black': (0, 0, 0),
    #         'gray': (0.5, 0.5, 0.5), 'grey': (0.5, 0.5, 0.5)
    #     }
    #
    #     if color_str in color_dict:
    #         return color_dict[color_str]
    #
    #     if color_str.startswith('#'):
    #         hex_color = color_str.lstrip('#')
    #         if len(hex_color) == 6:
    #             r = int(hex_color[0:2], 16) / 255.0
    #             g = int(hex_color[2:4], 16) / 255.0
    #             b = int(hex_color[4:6], 16) / 255.0
    #             return (r, g, b)
    #         elif len(hex_color) == 3:
    #             r = int(hex_color[0] * 2, 16) / 255.0
    #             g = int(hex_color[1] * 2, 16) / 255.0
    #             b = int(hex_color[2] * 2, 16) / 255.0
    #             return (r, g, b)
    #
    #     if color_str.startswith('rgb'):
    #         import re
    #         match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
    #         if match:
    #             r = int(match.group(1)) / 255.0
    #             g = int(match.group(2)) / 255.0
    #             b = int(match.group(3)) / 255.0
    #             return (r, g, b)
    #
    #     return (0, 0, 0)

    for i in range(rows):
        for j in range(cols):
            rgb_array[i, j] = (color_matrix[i][j][0], color_matrix[i][j][1], color_matrix[i][j][2])
            # rgb_array[i, j] = (0, 0, 0)

    return rgb_array

def xpm_to_image_pil(xpm_file, output_file='output.png'):
    """Read an XPM file with PIL."""
    try:
        img = Image.open(xpm_file)
        img.save(output_file)
        print(f"Image saved as: {output_file}")
        return img
    except Exception as e:
        print(f"PIL read failed: {e}")
        return parse_xpm_manually(xpm_file, output_file)

def parse_xpm_manually(xpm_file, output_file='output.png', ):
    """Parse an XPM file manually."""
    with open(xpm_file, 'r') as f:
        lines = f.readlines()
    
    start = -1
    for i, line in enumerate(lines):
        if 'static char' in line:
            start = i + 1
            break
    
    if start == -1:
        raise ValueError("No XPM data found")
    
    header = lines[start].strip().strip('",')
    width, height, colors, chars_per_pixel = map(int, header.split())
    
    color_map = matplotlib.cm.get_cmap('plasma', 100)
    color_value = color_map.colors
    color_table = {}
    for i in range(start + 1, start + 1 + colors):
        line = lines[i].strip().strip('",')
        if not line:
            continue
        char = line[:chars_per_pixel]
        color_def = line[chars_per_pixel:].strip()
        
        if 'c #' in color_def:
            color = color_def.split('#')[1][:6]
            if len(color) == 3:
                color = ''.join([c*2 for c in color])
            color_table[char] = f'#{color}'
        elif 'c None' in color_def or 'c none' in color_def:
            color_table[char] = None
    color_i = 0
    for key in color_table:
        color_table[key] = [float(color_value[color_i][0]), float(color_value[color_i][1]), float(color_value[color_i][2])]
        # color_table[key] = color_value[color_i][0:3]
        color_i = color_i + 1
    
    pixels = []
    for i in range(start + 1 + colors + 6, start + 1 + colors + 6 + height):
        line = lines[i].strip().strip('",')
        row = []
        for j in range(0, width * chars_per_pixel, chars_per_pixel):
            char = line[j:j + chars_per_pixel]
            default_value = color_table['KB']
            row.append(color_table.get(char, default_value))
        pixels.append(row)

    rgb_array = color_matrix_to_image(pixels)
    # rgb_array = pixels
    fig, ax = plt.subplots(figsize=(20, 16))
    # im = ax.imshow(rgb_array)
    im = ax.matshow(rgb_array, interpolation='nearest', cmap='plasma')
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    x_scale = (0, 408)
    y_scale = (408, 0)
    x_min, x_max = x_scale
    y_min, y_max = y_scale
    num_ticks = 16
    x_pixel_ticks = np.linspace(0, width - 1, num_ticks)
    y_pixel_ticks = np.linspace(0, height - 1, num_ticks)
    x_labels = np.linspace(x_min, x_max, num_ticks)
    y_labels = np.linspace(y_min, y_max, num_ticks)
    # ax.set_xticks(x_pixel_ticks)
    # ax.set_xticklabels(x_labels, rotation=45)
    # ax.set_yticks(y_pixel_ticks)
    # ax.set_yticklabels(y_labels)
    def format_labels(values, decimals=2):
        range_val = np.max(values) - np.min(values)
        if range_val > 100:
            decimals = 0
        elif range_val > 10:
            decimals = 1
        elif range_val > 1:
            decimals = 2
        else:
            decimals = 3

        return [f"{v:.{decimals}f}" for v in values]

    x_labels_formatted = format_labels(x_labels)
    y_labels_formatted = format_labels(y_labels)

    ax.set_xticks(x_pixel_ticks)
    ax.set_xticklabels(x_labels_formatted, rotation=45, fontsize=25)

    ax.set_yticks(y_pixel_ticks)
    ax.set_yticklabels(y_labels_formatted, fontsize=25)

    ax.set_xlabel('Residue Index', fontsize=28)
    ax.set_ylabel('Residue Index', fontsize=28)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.colorbar(im, ax=ax, pad=0.01)
    # cbar = plt.colorbar(im, ax=ax, pad=0.01, cmap='plasma')
    # plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Image saved to: {output_file}")

    plt.show()

    # from PIL import Image
    # img = Image.new('RGB', (width, height))
    # pixels_img = img.load()
    #
    # for y in range(height):
    #     for x in range(width):
    #         color = pixels[y][x]
    #         if color and color.startswith('#'):
    #             hex_color = color.lstrip('#')
    #             if len(hex_color) == 6:
    #                 r = int(hex_color[0:2], 16)
    #                 g = int(hex_color[2:4], 16)
    #                 b = int(hex_color[4:6], 16)
    #                 pixels_img[x, y] = (r, g, b)
    #             elif len(hex_color) == 3:
    #                 r = int(hex_color[0]*2, 16)
    #                 g = int(hex_color[1]*2, 16)
    #                 b = int(hex_color[2]*2, 16)
    #                 pixels_img[x, y] = (r, g, b)
    #
    # img.save(output_file)
    return ax

xpm_to_image_pil('mean.xpm', 'output.png')