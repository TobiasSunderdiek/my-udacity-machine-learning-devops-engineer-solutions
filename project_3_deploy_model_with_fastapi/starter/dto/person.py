from pydantic import BaseModel, Field


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    edcucation: str
    education_num: int = Field(None, alias='education-num')
    marital_status: str = Field(None, alias='marital_status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(None, alias='capital-gain')
    capital_loss: int = Field(None, alias='capital-loss')
    hours_per_week: int = Field(None, alias='hours_per_week')
    native_country: str = Field(None, alias='native-country')
    