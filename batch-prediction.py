from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import *


settings = EnvironmentSettings.new_instance().use_blink_planner().build()
exec_env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(exec_env, environment_settings=settings)

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
    user_id INT,
    movie_id INT
) WITH (
    'connector' = 'print'
)
"""

t_env.execute_sql(SOURCE_DDL)
t_env.execute_sql(SINK_DDL)
t_env.execute_sql(
    "INSERT INTO sink SELECT user_id, movie_id FROM source"
).wait()
