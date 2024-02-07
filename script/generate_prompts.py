# Read the content file
with open("./content_template.txt", "r") as content_file:
    content_list = content_file.read().splitlines()

with open("./style_template.txt", "r") as style_file:
    style_list = style_file.read().splitlines()

from itertools import product

# Generate the prompt.txt file with all combinations of content and style
content_style_combinations = list(product(content_list, style_list))


# Generate the prompt.txt file with unique names for each combination
with open("./prompt.txt", "w") as prompt_file, \
    open("./content.txt", "w") as content_file, \
    open("./style.txt", "w") as style_file:
    
    for idx, combination in enumerate(content_style_combinations, start=1):
        content, style = combination
        prompt_file.write(f"{content.replace(' ', '_')}_in_{style.replace(' ', '_')}\n")
        content_file.write(f"{content}\n")
        style_file.write(f"{style}\n")
