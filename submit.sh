echo "Start"

tag=aicregistry:5000/${USER}:adapterexp

echo "ghjdfi"

docker build ./docker           \
  --tag ${tag}                  \
  --build-arg USER_ID=$(id -u)  \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}      \
  --network=host

docker push ${tag}

runai submit            \
    --backoff-limit 0   \
    --name ${1}         \
    --image ${tag}      \
    --gpu 2            \
    --project ${USER}   \
    --volume /nfs:/nfs  \
    --large-shm         \
    --run-as-user       \
    --command -- ${@:2} \


