from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import os
import findspark

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-8-openjdk-amd64'
findspark.init(spark_home="/opt/spark-2.4.4/")


conf = SparkConf(). \
    setAppName('main'). \
    setMaster('local[*]'). \
    set('spark.yarn.appMasterEnv.PYSPARK_PYTHON', '~/anaconda3/bin/python'). \
    set('spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON', '~/anaconda3/bin/python'). \
    set('spark.executor.memoryOverhead', '16g'). \
    set('spark.sql.codegen', 'true'). \
    set('spark.yarn.executor.memory', ' 16g'). \
    set('spark.dynamicAllocation.maxExecutors', '4'). \
    set('spark.driver.maxResultSize', '0') .\
    set('spark.driver.memory', '4g') .\
    set('spark.execution.arrow.enabled', 'true')


spark = SparkSession.builder. \
    appName("main"). \
    config(conf=conf). \
    getOrCreate()

