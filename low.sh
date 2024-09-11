#!/bin/bash

# Define the command to execute
command1="python randomTrips.py -n VictoriaMap.net.xml -r ./lowdensity/lowRound"
command2=".rou.xml -e 1000 -p 20 --random"

round_counter=0

while ((round_counter < 10))
do
    # Concatenate the command with the round number
    command="${command1}${round_counter}${command2}"
    
    # Execute the command
    echo "Executing command: $command"
    #eval $command
    $command
    # Increment the round counter
    round_counter=$((round_counter + 1))
done