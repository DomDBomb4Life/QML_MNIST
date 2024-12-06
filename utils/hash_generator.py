# File: utils/hash_generator.py
import hashlib
import json

def generate_config_hash(params):
    """
    Generate a stable hash from a dictionary of parameters.
    Steps:
    1. Sort keys to ensure consistent ordering.
    2. Dump to JSON string with sorted keys.
    3. Hash the resulting string with SHA256 (or MD5).
    """
    # Convert params to a sorted JSON string for consistency
    sorted_str = json.dumps(params, sort_keys=True)
    # Create a SHA256 hash of the string
    hash_obj = hashlib.sha256(sorted_str.encode('utf-8')).hexdigest()
    return hash_obj[:12]  # Shorten the hash for directory naming convenience