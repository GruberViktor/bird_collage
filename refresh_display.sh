#!/bin/bash

cd /home/$USER/bird_collage
uv run python main.py
cd /home/$USER/inky
uv run examples/spectra6/image.py --file ~/bird_collage/collage.png --saturation 0.6