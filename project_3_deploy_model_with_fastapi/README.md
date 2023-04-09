## My solution to project 3 `Deploying a ML Model to Cloud Application Platform with FastAPI`

# My github project

If not viewing this on github, find my project here: https://github.com/TobiasSunderdiek/my-udacity-machine-learning-devops-engineer-solutions

# Starter code

Udacity provides starter code for this project, which can be found here: https://github.com/udacity/nd0821-c3-starter-code

As mentioned in the instructions of this course, starter code is downloaded, this new project folder is created and setup with git.

Therefore, my solution code is based on this starter code, all credits to the authors of the course.

License for this project: see Udacity License here: https://github.com/udacity/nd0821-c3-starter-code/blob/master/LICENSE.txt

# Setup

- `conda create -n project3 "python=3.8" --file starter/requirements.txt -c conda-forge`

- `conda activate project3`

# Usage

- run the tests with `pytest` (or with logging: `pytest --log-cli-level=DEBUG`)

- run `python src/sanitycheck.py` for a check of functionality to meet course specifications

- run `python src/ml/train_model.py ` to re-train the model (trained model is provided here: `src/model`)