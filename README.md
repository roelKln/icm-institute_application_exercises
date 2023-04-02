# ICM-institute application exercises

*Author: Roel Klein*

**Note**: developed and tested on Windows. I tried to make it as much OS independent, but some minor changes may need to be made for Mac/Linux compatability. 

- [ICM-institute application exercises](#icm-institute-application-exercises)
  - [Installation](#installation)
  - [Exercise answers / usage](#exercise-answers--usage)
    - [Exercise 1:](#exercise-1)
    - [Exercise 2:](#exercise-2)
    - [Exercise 3:](#exercise-3)


## Installation
1. Clone this repository:
```
git clone https://github.com/roelKln/icm-institute_application_exercises.git
```
2. Create new conda environment:
```shell
conda create --name icm-application python=3.9 pip
conda activate icm-application
```
3. Install correct Pytorch depending on OS and CUDA version, following the instructions [here](https://pytorch.org/get-started/locally/).
4. Install other packages:
```shell
pip install -r requirements.txt
```


## Exercise answers / usage
### Exercise 1:
* The training script is ```train.py```, run ```python train.py --help``` to see usage of all command-line parameters.
*  By default, automatic mixed precision is enabled. Use flag `--no-mixed_precision` to disable.
*  View results of a trained model on the validation and test set in ``` visualize.ipynb ```

### Exercise 2:
* To finetune only the last layer of a model, use:
```
python train.py -ft -c "path\to\trained_model.ckpt" 
```

### Exercise 3:
* By default, we loop through the training data once per epoch. We can also choose the number of batches/steps per epoch as follows, such that training samples can get shown multiple times per epoch, but with different data augmentation:
```
python train.py -s {number of steps per epoch}
```
