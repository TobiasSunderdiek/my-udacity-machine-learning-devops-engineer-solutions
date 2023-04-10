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

#####- `cd project_3_deploy_model_with_fastapi/`

- run the tests with ` python -m pytest project_3_deploy_model_with_fastapi -vv` (or add: `--log-cli-level=DEBUG` for logging)

- run `python -m project_3_deploy_model_with_fastapi.src.sanitycheck` with `project_3_deploy_model_with_fastapi/tests/test_main.py` as test file for a check of functionality to meet course specifications

- run `python -m project_3_deploy_model_with_fastapi.src.ml.train_model`

  - to re-train and save the model (already trained model is provided here: `src/model/lr_model.joblib` with `encoder.joblib` and `lb.joblib`)

  - to re-calculate metrics in `src/model/slice_output.txt`