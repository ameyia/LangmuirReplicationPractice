"""
search_and_populate_complexes.py

Two-stage tool:
  1. SEARCH: query RCSB's Search API for PDB entries matching filters
     (e.g. "exactly 2 distinct protein chains", resolution cutoff,
     optional keyword like a protein family or ligand name).
  2. RESOLVE: for each candidate PDB ID, query RCSB's Data API to
     look up the actual chain letters for each protein entity -
     this automates the "read COMPND by hand" step we did manually
     for 1BRS and 2PTC.

Output: a complexes.csv ready to feed into batch_extract_interfaces.py

Usage:
    python search_and_populate_complexes.py --keyword "trypsin inhibitor" --max_resolution 2.5 --limit 20

    # no keyword - just any two-protein-chain X-ray structure under 2.5A
    python search_and_populate_complexes.py --max_resolution 2.5 --limit 20

Requires:
    pip install requests --break-system-packages
"""

import argparse
import csv
import requests

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{}"
DATA_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{}/{}"


def search_candidates(keyword=None, max_resolution=3.0, limit=25, start=0):
    """
    Build and execute a Search API query for:
      - exactly 2 distinct polymer entities (roughly: 2 different molecules)
      - both entities are protein
      - X-ray structure with resolution <= max_resolution
      - optionally: full-text keyword match (e.g. a protein/family name)
    Returns a list of PDB IDs.
    """
    nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "equals",
                "value": 2
            }
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": max_resolution
            }
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.experimental_method",
                "operator": "exact_match",
                "value": "X-ray"
            }
        },
    ]

    if keyword:
        nodes.append({
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": keyword}
        })

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": limit},
            "results_content_type": ["experimental"],
        }
    }

    resp = requests.post(SEARCH_URL, json=query)
    resp.raise_for_status()
    data = resp.json()
    return [r["identifier"] for r in data.get("result_set", [])]


def resolve_chains(pdb_id):
    """
    For a given PDB ID, fetch entry info to find its polymer entity IDs,
    then fetch each entity's info to get its molecule name and chain
    letters (auth_asym_ids). Returns a list of (chain_letter, molecule_name)
    for protein entities only.
    """
    entry_resp = requests.get(DATA_ENTRY_URL.format(pdb_id))
    entry_resp.raise_for_status()
    entry_data = entry_resp.json()

    entity_ids = entry_data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])

    results = []
    for entity_id in entity_ids:
        ent_resp = requests.get(DATA_ENTITY_URL.format(pdb_id, entity_id))
        ent_resp.raise_for_status()
        ent_data = ent_resp.json()

        poly_type = ent_data.get("entity_poly", {}).get("rcsb_entity_polymer_type", "")
        if poly_type != "Protein":
            continue  # skip DNA/RNA/other entities

        chains = ent_data.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
        name = ent_data.get("rcsb_polymer_entity", {}).get("pdbx_description", "UNKNOWN")

        for chain in chains:
            results.append((chain, name))

    return results


def main():
    parser = argparse.ArgumentParser(description="Search RCSB for two-protein-chain complexes and auto-resolve chain letters.")
    parser.add_argument("--keyword", default=None, help="Optional full-text keyword, e.g. 'trypsin inhibitor' or 'collagen'")
    parser.add_argument("--max_resolution", type=float, default=2.5)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--start", type=int, default=0, help="Pagination offset - use 25, 50, 75... to get results beyond the first page")
    parser.add_argument("--out", default="complexes_auto.csv")
    args = parser.parse_args()

    print(f"Searching RCSB (keyword={args.keyword!r}, resolution<={args.max_resolution}, limit={args.limit}, start={args.start})...")
    candidates = search_candidates(args.keyword, args.max_resolution, args.limit, args.start)
    print(f"Found {len(candidates)} candidate entries.")

    rows = []
    for i, pdb_id in enumerate(candidates, 1):
        print(f"[{i}/{len(candidates)}] Resolving chains for {pdb_id}...", end=" ")
        try:
            chain_info = resolve_chains(pdb_id)
            protein_chains = chain_info  # already filtered to protein-only

            if len(protein_chains) < 2:
                print(f"SKIP (only {len(protein_chains)} protein chain(s) found - not a clean 2-chain case)")
                continue

            # Use the first chain of the first two distinct molecule names found.
            # (If a molecule has multiple copies, e.g. "A, B, C", this picks one representative.)
            seen_names = {}
            for chain, name in protein_chains:
                if name not in seen_names:
                    seen_names[name] = chain
                if len(seen_names) == 2:
                    break

            if len(seen_names) < 2:
                print(f"SKIP (only 1 distinct protein molecule - likely a homodimer, not a binding pair)")
                continue

            names = list(seen_names.keys())
            chain_a, chain_b = seen_names[names[0]], seen_names[names[1]]
            rows.append({
                "pdb_id": pdb_id, "chain_a": chain_a, "chain_b": chain_b,
                "molecule_a": names[0], "molecule_b": names[1],
            })
            print(f"OK: {chain_a} ({names[0][:30]}) + {chain_b} ({names[1][:30]})")

        except Exception as e:
            print(f"FAILED: {e}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pdb_id", "chain_a", "chain_b", "molecule_a", "molecule_b"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} complexes written to {args.out}")
    print("Review molecule_a/molecule_b columns before trusting this blindly -")
    print("drop the last two columns before feeding into batch_extract_interfaces.py.")


if __name__ == "__main__":
    main()