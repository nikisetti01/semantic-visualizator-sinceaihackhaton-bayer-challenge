# hse-mining-frontend

## Installation for local development

1. (Optional, but highly recommended) Initiate a new virtual env e.g.: `conda create -n hse python=3.13`

2. Install the project and all dependencies by `pip install -e .`

3. Assume the correct aws role using `aws-sso`.

4. To run the Streamlit app, use the following command in your terminal:
`streamlit run hse-mining-frontend/frontend.py` This will start a local server, and you can access
the frontend by navigating to http://localhost:8501 in your web browser.

## Installation for local developement (docker)

The recommended practice is to use the provided scripts:

1. Run `utils/build.sh`
2. Run `utils/run.sh`

The scripts will ensure the active aws role (assumed via aws-sso as explained above) will get passed
properly inside the running container. In addition, see remark in the Environment section about
handling the `AWS_DEFAULT_REGION` flag properly in context of local Docker runtimes.

## Project structure

The main project can be found under `hse-mining-frontend` and is organized under a set of distinct 
modules as follows:

* `frontend.py` contains the main wireframe for UI and serves as the starting point of the app
* `components.py` extends the frontend and is home to the largest main UI components and their behavior
* `awsconnection.py` contains means to communicate with the chosen backend system for the RAG 
application - in this case, AWS Athena
* `interactions.py` is intended to store all the functionalities where interaction with LLM API is
required
* `transformations.py` contains means for in-place data processing, before/between/after actual calls
to LLM or the backend
* `filters.py` contains static metadata which can be used for filtering the initial resultset of
the RAG

## Environment

`python-dotenv` is used for environment variable management during local development. Configure
your `.env` file e.g. as follows:

```
LLM_TOKEN=<Generate a personal or machine CWID token according to MGA documentation>
QUERY_ASSISTANT_ID=<Redacted from external access>
ANALYSIS_ASSISTANT_ID=<Redacted from external access>
DATABASE=<Redacted from external access>
WORKGROUP=<Redacted from external access>
S3_OUTPUT=<Redacted from external access>
DATA_SOURCE=<Redacted from external access>
AWS_DEFAULT_REGION=<Redacted from external access>
ATHENA_ACCESS_KEY=<Your access key to Athena> # Optional - intended primarily to use in production deployments
ATHENA_SECRET_ACCESS_KEY=<Your secret access key to Athena> # Optional - intended primarily to use in production deployments
```

`AWS_DEFAULT_REGION` is normally parsed automatically from your local AWS runtime, but there's a known
issue where the region information is not correctly passed to container when running under Docker.
This can be fixed by explicitly setting this `.env` variable and running the container with the
provided utility script in `utils/run.sh`, which uses the `--env-file` flag to override the values
provided at image build time.

## LLM Tokens

Tokens can be personal tokens, or they can be tied to a Machine CWID. Visit the respective documentation for instructions:

- <Redacted from external access>
- <Redacted from external access>
