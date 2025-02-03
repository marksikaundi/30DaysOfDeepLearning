#!/usr/bin/env python3
import os

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_folders():
    project_dir = os.getcwd()
    expected_folders = [f"{i:02d}" for i in range(1, 31)]
    existing_folders = [f for f in os.listdir(project_dir) if os.path.isdir(f)]

    missing_folders = []
    for folder in expected_folders:
        if folder not in existing_folders:
            missing_folders.append(folder)

    if not missing_folders:
        print(f"\n{GREEN}{BOLD}🎉 Congratulations! 🎉{RESET}")
        print(f"{GREEN}All folders from 01 to 30 are present in the directory!{RESET}")
        print(f"{GREEN}Great job keeping your project organized! 🌟{RESET}")
    else:
        print(f"\n{RED}{BOLD}⚠️  Some folders are missing:{RESET}")
        print(f"{RED}Missing folders: {', '.join(missing_folders)}{RESET}")
        print(f"\n{YELLOW}Found {len(existing_folders)} folders out of {len(expected_folders)} expected folders.{RESET}")

def main():
    print(f"{BOLD}Checking project folders...{RESET}")
    check_folders()

if __name__ == "__main__":
    main()
