You can use run.py to run all of the examples at once, or a specific folder, or a specific config file.

# Run every example (flax is ignored on windows)

python run.py

# Run every stable diffusion example

python run.py stablediffusion


# Run the stablediffusion/basic example

python run.py stablediffusion/basic


# Run the stablediffusion/animations/kitten-config.txt configuration

python run.py stablediffusion/animations/kitten-config.txt


# Pass arguments to dgenerate, run all examples with debuging output

python run.py -v

# Passing arguments works for specific examples as well

python run.py stablediffusion/basic -v
python run.py stablediffusion/animations/kitten-config.txt -v


