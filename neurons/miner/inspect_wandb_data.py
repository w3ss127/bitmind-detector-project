import json
import wandb
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any


def query_wandb_runs(project_name, entity_name=None, start_date=None, end_date=None, filters=None):
    """
    Query runs from a W&B project within a specified date range.
    
    Params:
        project_name : str
            Name of the W&B project to query
        entity_name : str, optional
            Name of the W&B entity (user or team)
        start_date : str or datetime, optional
            Start date for filtering runs (format: 'YYYY-MM-DD' if string)
        end_date : str or datetime, optional
            End date for filtering runs (format: 'YYYY-MM-DD' if string)
        filters : dict, optional
            Additional filters to apply (see W&B API documentation)
        
    Returns:
        runs : list
            List of W&B Run objects
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
    api = wandb.Api()
    
    filter_dict = filters or {}
    date_filters = []
    if start_date:
        date_filters.append({"created_at": {"$gte": start_date.isoformat()}})
    if end_date:
        date_filters.append({"created_at": {"$lte": end_date.isoformat()}})
        
    if date_filters:
        if len(date_filters) > 1:
            filter_dict = {"$and": [filter_dict, {"$and": date_filters}]} if filter_dict else {"$and": date_filters}
        else:
            filter_dict = {"$and": [filter_dict, date_filters[0]]} if filter_dict else date_filters[0]
    
    runs = api.runs(f"{entity_name}/{project_name}" if entity_name else project_name, filters=filter_dict)
    return runs

if __name__ == "__main__":
    # Dictionary to store all the data
    wandb_data = {
        "runs_count": 0,
        "runs_data": []
    }
    
    runs = query_wandb_runs(
        project_name="subnet-34-validator",
        entity_name="bitmindai",
        start_date="2025-07-10",
        end_date="2025-07-12",
    )

    print(f"Found {len(runs)} runs")
    wandb_data["runs_count"] = len(runs)

    for run_idx, run in enumerate(runs):
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "run_state": run.state,
            "created_at": run.created_at if run.created_at else None,
            "history_data": []
        }
        
        print(f"\nProcessing run {run_idx + 1}/{len(runs)}: {run.name}")
        
        for row_idx, row in enumerate(run.scan_history(page_size=100)):
            row_data = {
                "row_index": row_idx,
                "available_attributes": [],
                "example_values": {}
            }
            
            print(f"available row attributes:")
            for key in row.keys():
                if not key.startswith('_'):
                    print(key)
                    row_data["available_attributes"].append(key)

            print("Example values:")
            example_values = {}
            
            # Capture example values for key fields
            key_fields = [
                'results.miner_uids',
                'results.miner_hotkeys', 
                'results.metrics',
                'results.rewards',
                'results.scores'
            ]
            
            for field in key_fields:
                if field in row:
                    value = row[field]
                    print(f'\t{field}: {value}')
                    example_values[field] = str(value) if value is not None else None
                else:
                    print(f'\t{field}: Not found')
                    example_values[field] = None
            
            row_data["example_values"] = example_values
            run_data["history_data"].append(row_data)
            
            # Only process first row for each run (as in original code)
            break
        
        wandb_data["runs_data"].append(run_data)
        # Only process first run (as in original code)
        break

    # Save to JSON file
    output_file = "wandb.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wandb_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Data saved to {output_file}")
    print(f"📊 Summary: {len(runs)} runs processed, data structure saved to JSON")