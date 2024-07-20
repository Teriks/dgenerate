# helper script which condenses all repeated local links in README.rst
# to hard references at the top of the RST file, this prevents packaging
# and RST rendering errors.

import os
import re
from collections import defaultdict

# Change to the parent directory of the script location
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def find_and_condense_links(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # regex pattern to match links in rst format starting with http or https
    link_pattern = re.compile(r'`([^`]+) <(https?://[^>]+)>`_')

    links = defaultdict(list)

    # find all links in the file and populate the dictionary
    for line in lines:
        for match in re.findall(link_pattern, line):
            title, url = match
            links[title].append(url)

    # find duplicates
    duplicates = {title: urls for title, urls in links.items() if len(urls) > 1}

    if not duplicates:
        print("No duplicate links found.")
        return

    # prepare the condensed links and reference map
    condensed_links = []
    ref_map = {}
    for title, urls in duplicates.items():
        for idx, url in enumerate(set(urls), 1):
            ref_name = f"{title.replace(' ', '_')}_{idx}"
            condensed_links.append(f".. _{ref_name}: {url}")
            ref_map[url] = ref_name

    # replace local usages of the URLs with the consolidated links
    new_lines = []
    for line in lines:
        new_line = line
        for match in re.findall(link_pattern, line):
            title, url = match
            if url in ref_map:
                new_ref = f"`{title} <{ref_map[url]}_>`_"
                new_line = new_line.replace(f"`{title} <{url}>`_", new_ref)
        new_lines.append(new_line)

    # write the condensed links at the top of the file and update the file content
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(condensed_links) + '\n\n')
        file.writelines(new_lines)

    print("Duplicate links condensed, and local usages updated.")


# Run the function on README.rst
find_and_condense_links('README.rst')
