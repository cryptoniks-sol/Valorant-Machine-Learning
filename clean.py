import re

with open("dirty.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

with open("dirty.txt", "w", encoding="utf-8") as file:
    for line in lines:
        line = line.rstrip()
        if line and not re.match(r'\s*#', line):
            file.write(line + "\n")
