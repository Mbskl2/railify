def generate_svg_from_lines(line_segments, width=500, height=500):
    """
    Create an SVG representation from a list of line segments.

    :param line_segments: List of tuples (x1, y1, x2, y2)
    :param width: Width of the SVG canvas
    :param height: Height of the SVG canvas
    :return: SVG string
    """
    svg_header = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    svg_lines = ""

    for (x1, y1, x2, y2) in line_segments:
        svg_lines += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="blue" stroke-width="20" />\n'

    svg_footer = '</svg>'
    return svg_header + svg_lines + svg_footer



def save_svg(svg_content, output_file):
    # Specify the output file path

    # Write the SVG content to a file
    with open(output_file, 'w') as file:
        file.write(svg_content)

    print(f'SVG content saved to {output_file}')

    return output_file
