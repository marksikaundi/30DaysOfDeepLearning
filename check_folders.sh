#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Function to draw progress bar
draw_progress_bar() {
    local width=50
    local percentage=$1
    local filled=$(printf "%.0f" $(echo "scale=2; $width * $percentage / 100" | bc))
    local empty=$((width - filled))

    printf "${BLUE}["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] ${percentage}%%${NC}\r"
}

echo -e "${BOLD}Checking for folders 01-30...${NC}\n"

# Initialize counter for existing folders
existing_folders=0
missing_folders=()

# Check each folder from 01 to 30 with progress bar
for i in $(seq -w 1 30)
do
    if [ -d "$i" ]; then
        ((existing_folders++))
    else
        missing_folders+=("$i")
    fi

    # Calculate and show progress
    progress=$((existing_folders * 100 / 30))
    draw_progress_bar $progress
done

echo -e "\n"

# Check if all folders exist
if [ ${#missing_folders[@]} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}🎉 Congratulations! 🎉${NC}"
    echo -e "${GREEN}All folders from 01 to 30 are present!${NC}"
    echo -e "${GREEN}Great job keeping your project organized! 🌟${NC}"
else
    echo -e "${RED}${BOLD}⚠️  Missing folders:${NC}"
    echo -e "${RED}${missing_folders[@]}${NC}"
    echo -e "\n${YELLOW}Found $existing_folders folders out of 30 expected folders.${NC}"
    echo -e "${YELLOW}Progress: $(( (existing_folders * 100) / 30 ))%${NC}"
fi
