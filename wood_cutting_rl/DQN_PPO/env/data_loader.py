# data_loader.py
import csv
from .constants import DEFAULT_STOCKS, DEFAULT_PRODUCTS

def load_stock_list(file_path):
    if not file_path:
        return DEFAULT_STOCKS
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            return [(int(row['width']), int(row['height'])) for row in reader]
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading stock file {file_path}: {e}. Using default.")
        return DEFAULT_STOCKS

def load_product_list(file_path):
    if not file_path:
        return DEFAULT_PRODUCTS
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            return [(int(row['width']), int(row['height']), int(row['quantity'])) for row in reader]
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading product file {file_path}: {e}. Using default.")
        return DEFAULT_PRODUCTS