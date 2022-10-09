#!/bin/sh

export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=2
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export LOCAL_PATH=/home/sim/Documents/medusa

echo "Script executed from: ${PWD}"

${SPARK_HOME}/sbin/start-master.sh;${SPARK_HOME}/sbin/start-worker.sh -c $CORES_PER_WORKER -m 3G ${MASTER}

echo "Remove any old artifacts"
rm -Rf ${LOCAL_PATH}/model/saved/segmentation_model.h5

echo "Train..."
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
src/segmentation.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--model_dir ${LOCAL_PATH}/model/checkpoint/ \
--export_dir ${LOCAL_PATH}/model/saved/ \
--epochs 2

echo "Confirm model"
ls -lR ${LOCAL_PATH}/model/saved
ls -lR ${LOCAL_PATH}/model/checkpoint/

echo "Shutdown the Spark Standalone cluster"
${SPARK_HOME}/sbin/stop-worker.sh; ${SPARK_HOME}/sbin/stop-master.sh