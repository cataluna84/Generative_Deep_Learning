#!/usr/bin/env python3
"""
Migration script for GAN Camel Run Folder.

Migrates existing flat-structure files from v1/run/gan/0001_camel/
to v1/run/gan/0001_camel/012/ (12th experiment run).
"""
import os
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
SOURCE_FOLDER = Path("v1/run/gan/0001_camel")
TARGET_RUN_ID = "012"
TARGET_FOLDER = SOURCE_FOLDER / TARGET_RUN_ID

# Files and folders to migrate (top-level items in source folder)
ITEMS_TO_MIGRATE = [
    # Model files
    "discriminator.h5",
    "discriminator.keras",
    "generator.h5",
    "generator.keras",
    "model.h5",
    "model.keras",
    "obj.pkl",
    "params.pkl",
    # Plots
    "accuracy_plot.png",
    "loss_plot.png",
    "lr_schedule_plot.png",
    "training_summary.png",
    # Directories
    "images",
    "viz",
    "weights",
]


def migrate():
    """Execute the migration."""
    print(f"Migrating files from {SOURCE_FOLDER} to {TARGET_FOLDER}")
    print("=" * 60)
    
    # Create target folder
    if not TARGET_FOLDER.exists():
        TARGET_FOLDER.mkdir(parents=True)
        print(f"✓ Created target folder: {TARGET_FOLDER}")
    else:
        print(f"⚠ Target folder already exists: {TARGET_FOLDER}")
    
    # Count successful migrations
    migrated_files = 0
    migrated_dirs = 0
    
    for item in ITEMS_TO_MIGRATE:
        source = SOURCE_FOLDER / item
        dest = TARGET_FOLDER / item
        
        if not source.exists():
            print(f"  ⊘ Skipping (not found): {item}")
            continue
        
        if dest.exists():
            print(f"  ⊘ Skipping (already exists): {item}")
            continue
        
        if source.is_dir():
            # Copy directory
            shutil.copytree(source, dest)
            migrated_dirs += 1
            print(f"  ✓ Copied directory: {item}")
        else:
            # Copy file
            shutil.copy2(source, dest)
            migrated_files += 1
            print(f"  ✓ Copied file: {item}")
    
    print("=" * 60)
    print(f"Migration complete: {migrated_files} files, {migrated_dirs} directories")
    print(f"Target folder: {TARGET_FOLDER}")
    
    return migrated_files, migrated_dirs


if __name__ == "__main__":
    migrate()
