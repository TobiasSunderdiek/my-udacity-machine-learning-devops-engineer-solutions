## My solution to project 3 `Deploying a ML Model to Cloud Application Platform with FastAPI`

# My github project

If not viewing this on github, find my project here: https://github.com/TobiasSunderdiek/my-udacity-machine-learning-devops-engineer-solutions/tree/main/project_3_deploy_model_with_fastapi

# Starter code

Udacity provides starter code for this project, which can be found here: https://github.com/udacity/nd0821-c3-starter-code

As mentioned in the instructions of this course, starter code is downloaded, this new project folder is created and setup with git.

Therefore, my solution code is based on this starter code, all credits to the authors of the course.

License for this project: see Udacity License here: https://github.com/udacity/nd0821-c3-starter-code/blob/master/LICENSE.txt

# Setup

- `conda create -n project3 "python=3.8" --file project_3_deploy_model_with_fastapi/requirements.txt -c conda-forge` in root of repository

- `conda activate project3`

# Usage

- run the tests with `python -m pytest project_3_deploy_model_with_fastapi -vv` (add `--log-cli-level=DEBUG` for logging)

- run `python -m project_3_deploy_model_with_fastapi.src.check.sanitycheck` and answer path question with `project_3_deploy_model_with_fastapi/tests/test_main.py` as test file for a check of functionality to meet course specifications

- run `python -m project_3_deploy_model_with_fastapi.src.ml.train_model`

  - to re-train and save the model (already trained model is provided here: `project_3_deploy_model_with_fastapi/src/model/lr_model.joblib` with `encoder.joblib` and `lb.joblib`)

  - to re-calculate metrics in `project_3_deploy_model_with_fastapi/src/model/slice_output.txt`

- run `uvicorn project_3_deploy_model_with_fastapi.src.main:app` to start REST-Endpoints locally on `http://127.0.0.1:8000`

  - see docs here: `http://127.0.0.1:8000/docs`

- to request app deployed on render or similar service

  - configure `RENDER_APP_URL` in `project_3_deploy_model_with_fastapi/src/check/request_render.py`

  - run `python -m project_3_deploy_model_with_fastapi.src.check.request_render`

- run `flake8 project_3_deploy_model_with_fastapi/` to lint locally

# Deploy on Render.com

- deploy repository as `Web Service` with specific configuration:

  - Repository: `https://github.com/TobiasSunderdiek/my-udacity-machine-learning-devops-engineer-solutions`

  - Branch: `main`

  - Build Command: `pip install -r project_3_deploy_model_with_fastapi/requirements.txt`

  - Start Command: `uvicorn project_3_deploy_model_with_fastapi.src.main:app --host 0.0.0.0 --port 10000`

  - Auto-Deploy: `Yes`

# Model

See model card for more details: https://github.com/TobiasSunderdiek/my-udacity-machine-learning-devops-engineer-solutions/blob/main/project_3_deploy_model_with_fastapi/model_card.md

# GithubActions

- only if changes are made within this project 3, github actions are called

    - with one exception: if a tag is pushed, github actions are also called