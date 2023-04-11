import requests
import logging


RENDER_APP_URL = 'https://predict-salary-fhxt.onrender.com'

logging.basicConfig(level=logging.DEBUG)


def post_to_render():
    post_path = f"{RENDER_APP_URL}/predict_salary"
    person = "{\"age\":39,\
            \"workclass\": \"State-gov\",\
            \"fnlgt\":77516,\
            \"education\":\"Bachelors\",\
            \"education-num\":13,\
            \"marital-status\":\"Never-married\",\
            \"occupation\":\"Adm-clerical\",\
            \"relationship\":\"Not-in-family\",\
            \"race\":\"White\",\
            \"sex\":\"Male\",\
            \"capital-gain\":2174,\
            \"capital-loss\":0,\
            \"hours-per-week\":40,\
            \"native-country\":\"United-States\"}"
    logging.info(f"Post {person} to {post_path}")
    res = requests.post(post_path, data=person)
    logging.info(f"Response status code: {res.status_code}")
    logging.info(f"Response msg: {res.json()}")


if __name__ == '__main__':
    post_to_render()
