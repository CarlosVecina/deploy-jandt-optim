"""
J and T homework. Carlos Vecina Tebar 2022.
Deploying via Heroku and FastAPI.
The App: Marketplace contact engine optimization.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.requests import Request
import logging
from datetime import datetime
from fastapi import FastAPI, APIRouter, Depends

from .optims.optim_exp import OptimExp
from .optims.optim_nbinomial import OptimNegBinom
from .optims.optim_stoch_constraint import OptimStochConstraint


logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    evel=logging.DEBUG)

logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "predict",
        "description": "Operations with users. The **login** logic is also here.",
    },
    {
        "name": "predict-post",
        "description": "Manage items. So _fancy_ they have their own docs.",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
]

app = FastAPI(title="Job and Talent - Technical Interview",
    description="Endpoints for optimizing models",
    version="0.0.2",
    contact={
        "name": "Carlos Vecina",
        "url": "http://carlosvecina.es",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },)

## https://github.com/tiangolo/fastapi/issues/394
api_router = APIRouter()


class ModelParams(BaseModel):
    now: datetime
    deadline: datetime
    num_vacancies: int
    num_remaining_in_pool: int
    impacted_candidates_data: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{param1}/{param2}", tags=["predict"])
def predict(param1: int, param2: int):

    pred = {'prediction': int(param1 + param2), 'probability': param2/100}
    return pred


@api_router.post("/optim-exp/", tags=["optim-exp"])
def post_predict(params: ModelParams):
    finished, num_candidates_needed, callback_time_minutes = OptimExp(is_decay=False).invitation_logic_api(
        now=params.now,
        deadline=params.deadline,
        num_vacancies=params.num_vacancies,
        num_remaining_in_pool=params.num_remaining_in_pool,
        impacted_candidates_data=params.impacted_candidates_data
    )

    logger.info(f"POST RESP {bool(finished)}.")

    return bool(finished), int(num_candidates_needed), int(callback_time_minutes)


@api_router.post("/optim-nbinomial/", tags=["optim-nbinomial"])
def post_predict(params: ModelParams):
    finished, num_candidates_needed, callback_time_minutes = OptimNegBinom().invitation_logic_api(
        now=params.now,
        deadline=params.deadline,
        num_vacancies=params.num_vacancies,
        num_remaining_in_pool=params.num_remaining_in_pool,
        impacted_candidates_data=params.impacted_candidates_data
    )

    logger.info(f"POST RESP {bool(finished)}.")

    return bool(finished), int(num_candidates_needed), int(callback_time_minutes)


@api_router.post("/optim-stoch-constraint/", tags=["optim-stoch-constraint"])
def post_predict(params: ModelParams):
    finished, num_candidates_needed, callback_time_minutes = OptimStochConstraint(beta_mean=0.2, beta_var=0.001).invitation_logic_api(
        now=params.now,
        deadline=params.deadline,
        num_vacancies=params.num_vacancies,
        num_remaining_in_pool=params.num_remaining_in_pool,
        impacted_candidates_data=params.impacted_candidates_data
    )

    logger.info(f"POST RESP {bool(finished)}.")

    return bool(finished), int(num_candidates_needed), int(callback_time_minutes)


async def log_json(request: Request):
    logger.info(f"POST REQ {str(await request.json())}.")
    print(await request.json())

app.include_router(api_router, dependencies=[Depends(log_json)])