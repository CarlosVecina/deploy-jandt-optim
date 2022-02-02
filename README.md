<h1 align="center">Technical Homework Senior Data Scientist</h1>
<h2 align="center">Deployment FastAPI via Heroku</h2>

## 📝 Table of Contents

- [About the Optims](#optims)
- [About the Deployment](#deploy)
- [Geting Started](#getting_started)

<br>

## 👨‍🔬 About the Optims<a name = "optims"></a>

All the logic falls under the app folder. To the date, we have develope 3 optimizatión objects. `OptimExp`, `OptimNegBinomial` and `OptimStochConstraint`

They make use of Bayesian optimization, distribution boostraping and linear optimization to address the problem of a cold start agent, without data or defined environment to learn from. We should model not only the number of new candidates impacted each call from the pool, but the callback in minutes till the next one.

All of them have the `invitation_logic_api()` method, which receive the same input:

| Field | Description | Type | Example |
| :---: | :---: | :---: | :---: |
| `correlation_id` | Event id. Case{n case}_{n call} | string | "Case0_2" 
| `reference_date_time` | api call timestamp | string (timestamp YYYYMMDD HH:MM:SS) | "2021-01-02 03:12:27"
| `deadline` | job opening deadline timestamp | string (timestamp YYYYMMDD HH:MM:SS) | "2021-01-05 09:00:00"
| `num_vacancies` | job opening number of vacancies (constant) | int | 8
| `num_remaining_in_pool` | remaining pool workers after the invitations already sent | int | 98
| `impacted_candidates_data` | information about the impacted candidates. A list of dictionaries containing their current status and how long it took to receive the reply (so far) | list | [{"notification_status": "ir_rejected", "candidate_status": "not_in_ft", "time_to_respond_ir_minutes": 1 }]

<br>

## 🧐 About the Deployment<a name = "deploy"></a>


We have make available the optim engines throuhg API endpoints deployed in Heroku accessible URI: 

`https://jandt-technical-carlosvecina.herokuapp.com/`

You could find API documentation:

`https://jandt-technical-carlosvecina.herokuapp.com/docs`
`https://jandt-technical-carlosvecina.herokuapp.com/redoc`

Each  repo is the source to the automatic integration FastAPI -> Heroku. 
The app automatically deploys from this repo *main* branch.

You could access to the API `optim-stoch-constraint` endpoint in Python using:

```python
import requests
from datetime import datetime as dt

optim = 'optim-nbinomial'
url = f'https://jandt-technical-carlosvecina.herokuapp.com/{optim}/'

data = {'now':str(datetime.datetime(2021,11,1)),
        'deadline': str(datetime.datetime(2021,11,2)),
        'num_vacancies': 10,
        'num_remaining_in_pool':500,
        'impacted_candidates_data': [        {
            "notification_status": "ir_accepted",
            "candidate_status": "offer_accepted",
            "time_to_respond_ir_minutes": 48
        },
        {
            "notification_status": "ir_rejected",
            "candidate_status": "not_in_ft",
            "time_to_respond_ir_minutes": 11
        }]
       }

resp = requests.post(url, json=data)
print(resp.json())
```

<br>

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine.


### Installing

#### Requirements

- [Docker >= 17.05](https://www.python.org/downloads/release/python-381/)
- [Python >= 3.7](https://www.python.org/downloads/release/python-381/)
- [Poetry](https://github.com/python-poetry/poetry)

> **NOTE** - Run all commands from the project root

### Local development

### Poetry

Create the virtual environment and install dependencies with:

```shell
poetry install
```

See the [poetry docs](https://python-poetry.org/docs/) for information on how to add/update dependencies.

Spawn a shell inside the virtual environment with:

```shell
poetry shell
```

Start a development server locally:

```shell
poetry run uvicorn app.main:app --reload --host localhost --port 8000
```

API will be available at [localhost:8000/](http://localhost:8000/)

### Docker

Build images with:
```shell
docker build --tag poetry-project --file docker/Dockerfile .
```

The Dockerfile uses multi-stage builds to run lint and test stages before building the production stage.
If linting or testing fails the build will fail.

You can stop the build at specific stages with the `--target` option:

```shell
docker build --name poetry-project --file docker/Dockerfile . --target <stage>
```

For example we wanted to stop at the **test** stage:

```shell
docker build --tag poetry-project --file docker/Dockerfile --target test .
```

We could then get a shell inside the container with:

```shell
docker run -it poetry-project:latest bash
```


### Credits

- Poetry Docker Heroku buildpack [Michael Oliver](https://github.com/michael0liver/python-poetry-docker-example)



