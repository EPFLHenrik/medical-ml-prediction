def update_indices(original_indices, indices_to_drop):
    # Keep only items whose key is not in to_drop
    kept_items = [(k, v) for k, v in original_indices.items() if k not in indices_to_drop]
    
    # Sort by the old index, then reassign new indices sequentially
    kept_items.sort(key=lambda kv: kv[1])
    
    # Rebuild dictionary with new indices
    new_dict = {k: i for i, (k, _) in enumerate(kept_items)}
    
    return new_dict