from collections import defaultdict

# Initialize data structures
true_ids = []
false_ids = []
id_true_count = defaultdict(int)
all_seen_ids = set()

# Read and parse file
with open('tmp.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# Process lines in pairs
for i in range(0, len(lines), 2):
    bools = lines[i].split()[-3:]
    id_line = lines[i+1]
    
    try:
        id_val = int(id_line.split(':')[-1].strip())
    except ValueError:
        print(f"Could not parse ID on line {i+1}: {id_line}")
        continue
    
    all_seen_ids.add(id_val)

    bool1 = bools[0] == '1'
    if bool1:
        true_ids.append(id_val)
        id_true_count[id_val] += 1
    else:
        false_ids.append(id_val)

# Identify duplicates in true_ids
duplicate_true_ids = [id for id, count in id_true_count.items() if count > 1]

# Identify IDs that were only ever false
only_false_ids = [id for id in set(false_ids) if id_true_count[id] == 0]

# Output results
print("IDs where bool1 is True:", true_ids)
print("IDs where bool1 is False:", false_ids)
print("IDs with multiple True occurrences:", duplicate_true_ids)
print("IDs that were only ever False:", only_false_ids)
