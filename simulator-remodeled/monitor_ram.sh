#!/bin/bash

# Interval in seconds (30 minutes = 1800 seconds)
interval=1800
# Sampling frequency in seconds (check RAM every 5 seconds)
sample_freq=5

max_ram_used=0
start_time=$(date +%s)

echo "Monitoring maximum RAM usage every 30 minutes. Press Ctrl+C to stop."
echo "Timestamp                     Max RAM Used (MB) in Interval"
echo "----------------------------------------------------------"

while true; do
    # Get current used RAM in MB from 'free -m'
    # Looks for the line starting with "Mem:", then prints the 3rd field (used)
    current_ram_used=$(free -m | awk '/^Mem:/ {print $3}')

    # Update max if current usage is higher
    if [[ "$current_ram_used" -gt "$max_ram_used" ]]; then
        max_ram_used=$current_ram_used
    fi

    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # Check if 30 minutes have passed
    if (( elapsed_time >= interval )); then
        timestamp=$(date +"%Y-%m-%d %H:%M:%S")
        printf "%-30s %s MB\n" "$timestamp" "$max_ram_used"

        # Reset for the next interval
        max_ram_used=0
        start_time=$(date +%s)
    fi

    # Wait before the next sample
    sleep $sample_freq
done
