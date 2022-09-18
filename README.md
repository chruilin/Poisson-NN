# Poisson-NN
## A Poisson-NN embedded solver: Speed up solving the incompressible Navier-Stokes equations

### how to run:

python ./Poisson-NN1024x1024/train.py --dtype=float32

python ./Poisson-NN1024x1024/train.py --dtype=float64

python ./Poisson-NN1024x1024/test.py --test_case=k16TGV


python ./Poisson-NN2048x2048/train.py --dtype=float32

python ./Poisson-NN2048x2048/train.py --dtype=float64

python ./Poisson-NN2048x2048/test.py --test_case=k16TGV

python paperfiguresplot.py
