###  python environment
# Download Miniconda or Anaconda. Note: if you are using Anaconda on Windows
#   you will likely need some special environment installation steps. TO BE ADDED.

# Create the environment:
conda env create -f environment.yml
conda activate ENV_ML # to activate it

# Setup: download and extract dogs-vs-cats.zip
#  Put the training dataset is in multiple separate subdirectories such that
#  ./train/cat/  <-- contains the cat training images
#  ./train/dog/  <-- contains the dog training images
#  ... This is the way the ImageDataGenerator classifies the training set, though
#      there may be other ways to do it I'm not aware of.

# Train the model:
python model_train.py  # saves the model to catdog_cnn_model.h5
#  Gotcha: make sure when the training dataset is discovered, it sees exactly 2 classes,
#  or you will have invalid results later.

# Execute the model:
python model_test.py
# This looks at the first 50 test images. They are unlabeled.
# Another way to analyze is to look at the training images, or better yet,
# separate 10% of the training images before training the model


# TO DO:
#  - Document the procedure for installing on Anaconda on Windows
#  - I had significantly different results reported in the training accuracy and
#    loss function while training with the same parameters on Windows vs Linux.
#    Explore and verify this.

