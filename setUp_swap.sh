#!/bin/bash

# sefine swapfile size
SWAP_SIZE="16G"
SWAPFILE="/swapfile"

# create
echo "Creating a ${SWAP_SIZE} swapfile at ${SWAPFILE}..."
sudo fallocate -l $SWAP_SIZE $SWAPFILE

echo "Setting permissions for the swapfile..."
sudo chmod 600 $SWAPFILE

echo "Setting up swap space..."
sudo mkswap $SWAPFILE

echo "Enabling the swap..."
sudo swapon $SWAPFILE

# make changes permanent
echo "Updating /etc/fstab to make swap permanent..."
echo "$SWAPFILE none swap sw 0 0" | sudo tee -a /etc/fstab

#lastly, verify
echo "Swap space activated:"
sudo swapon --show
echo "System memory info:"
free -h
