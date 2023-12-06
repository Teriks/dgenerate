You can use run.py to run all of the examples at once, or a specific folder, or a specific config file.

# Run every example (flax is ignored on windows)
# Use python3 on linux

# See python run.py --help

python run.py

# Run every stable diffusion example

python run.py --paths stablediffusion

# Run every stable diffusion example, do not
# run examples that generate animations

python run.py --paths stablediffusion --skip-animations


# Run the stablediffusion/basic example

python run.py --paths stablediffusion/basic


# Run the stablediffusion/animations/kitten-config.txt configuration

python run.py --paths stablediffusion/animations/kitten-config.txt


# Pass arguments to dgenerate, run all examples with debugging output

python run.py -v

# Passing arguments works for specific examples as well

python run.py --paths stablediffusion/basic -v
python run.py --paths stablediffusion/animations/kitten-config.txt -v


# you can run library_usage examples but arguments
# will be ignored for them

python run.py --paths  library_usage/basic


