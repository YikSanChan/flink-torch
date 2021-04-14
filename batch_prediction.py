from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import *
from pyflink.table.udf import udf

import torch

from model_def import MatrixFactorization

# Load model

n_users, n_items = 943, 1682
model = MatrixFactorization(n_users, n_items)
model.load_state_dict(torch.load("model.pth"))
print(model)

# Define UDF


@udf(result_type=DataTypes.DOUBLE())
def predict(user, item):
    return model([user], [item]).item()


settings = EnvironmentSettings.new_instance().use_blink_planner().build()
exec_env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(exec_env, environment_settings=settings)

t_env.create_temporary_function("predict", predict)

SOURCE_DDL = """
CREATE TABLE source (
    user_id INT,
    movie_id INT,
    rating TINYINT,
    event_ms BIGINT
) WITH (
    'connector' = 'filesystem',
    'format' = 'csv',
    'csv.field-delimiter' = '\t',
    'path' = 'ml-100k/u1.test'
)
"""

SINK_DDL = """
CREATE TABLE sink (
    prediction DOUBLE
) WITH (
    'connector' = 'print'
)
"""

t_env.execute_sql(SOURCE_DDL)
t_env.execute_sql(SINK_DDL)
t_env.execute_sql(
    "INSERT INTO sink SELECT PREDICT(user_id, movie_id) FROM source"
).wait()
