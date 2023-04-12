from pydantic import BaseModel, Field


class Person(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias='education-num')
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')
    native_country: str = Field(..., alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'capital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }
        }
