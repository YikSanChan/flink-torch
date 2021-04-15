from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import *

from udf_def import predict


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
    user_id INT,
    topk_movies STRING
) WITH (
    'connector' = 'print'
)
"""

TRANSFORM_DML = """
INSERT INTO sink
WITH t1 AS (
    SELECT user_id, movie_id, PREDICT(user_id, movie_id) predicted_rating
    FROM source
),
t2 AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY predicted_rating DESC) as row_num
    FROM t1
),
topk AS (
    SELECT user_id, movie_id
    FROM t2
    WHERE row_num <= 5
)
SELECT user_id, LISTAGG(CAST(movie_id AS STRING)) topk_movies
FROM topk
GROUP BY user_id
"""

t_env.execute_sql(SOURCE_DDL)
t_env.execute_sql(SINK_DDL)
t_env.execute_sql(TRANSFORM_DML).wait()
