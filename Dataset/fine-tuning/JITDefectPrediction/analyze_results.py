import json
from collections import defaultdict

def process_results(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)

    clone_groups = defaultdict(list)
    inconsistent_clone_groups = dict()
    hash_set = set()

    for result in results:
        query_info = {
            "hash": result["query_hash"],
            "label": result["query_label"],
            "message": result["query_message"],
            "change": result["query_change"]
        }
        similar_changes = result["similar_changes"]

        query_hash = query_info['hash']
        if query_hash not in hash_set:
            hash_set.add(query_hash)
            if query_info['hash'] not in clone_groups.keys():
                clone_groups[query_hash].append(query_info)

        # Iterate over similar changes for each query
        for similar_change in similar_changes:
            clone_info = {
                "hash": similar_change["hash"],
                "label": similar_change["label"],
                "message": similar_change["message"],
                "change": similar_change["change"]
            }
            clone_hash = clone_info['hash']
            if clone_hash not in hash_set:
                hash_set.add(clone_hash)
                clone_groups[query_hash].append(clone_info)

    # Check for inconsistent labels within clone groups
    for query_hash, group in clone_groups.items():
        labels = set(info["label"] for info in group)
        if len(labels) > 1:
            inconsistent_clone_groups[query_hash] = group

    print(f"For {results_file}")
    print("Total clone groups:", len(clone_groups))
    print("Total samples contained in clone groups:", sum(len(group) for group in clone_groups.values()))
    print("Total inconsistent clone groups:", len(inconsistent_clone_groups))
    print("Total samples affected by inconsistent labels:",
          sum(len(group) for group in inconsistent_clone_groups.values()))

    results_file_name = results_file.split('/')[-1].split('.json')[0]
    project_name = results_file.split('/')[-2]
    # Save clone groups for further inspection
    with open(f'./data_SimCom/clone/{project_name}/{results_file_name}_clone_groups.json', 'w') as f:
        json.dump(clone_groups, f, indent=4)
    print(f"Clone groups for {results_file} are saved to {results_file_name}_clone_groups.json")

    with open(f'./data_SimCom/clone/{project_name}/{results_file_name}_inconsistent_clone_groups.json', 'w') as f:
        json.dump(inconsistent_clone_groups, f, indent=4)
    print(f"Inconsistent clone groups for {results_file} are saved to {results_file_name}_inconsistent_clone_groups.json")

    return clone_groups, inconsistent_clone_groups

for project in ['gerrit', 'go', 'jdt', 'openstack', 'platform', 'qt']:
# for project in ['gerrit']:
    print(f'Begin to process {project}')
    results_file = f'./data_SimCom/clone/{project}/similar_changes_results_{project}.json'
    clone_groups, inconsistent_clone_groups = process_results(results_file)
    results_file_filtered = f'./data_SimCom/clone/{project}/similar_changes_results_{project}_filtered.json'
    clone_groups_filtered, inconsistent_clone_groups_filtered = process_results(results_file_filtered)
    print('-' * 50)

