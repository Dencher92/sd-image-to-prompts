#!/bin/bash
echo "$@"
if [  "`id -un`" = "root" ]; then
    if [ ! -z "${HOST_USER_NAME}" ] && [ ! -z "${HOST_USER_ID}" ]; then
        USER_NAME=`id -un ${HOST_USER_ID} 2>/dev/null`
        if [ $? != 0 ]; then
            useradd ${HOST_USER_NAME} -u ${HOST_USER_ID} -m
            USER_NAME=`id -un ${HOST_USER_ID} 2>/dev/null`
            if [ $? != 0 ]; then
                echo "ERROR: Can't change user to ${HOST_USER_NAME}"
            fi
        fi
        exec su ${USER_NAME} -c "$@"
    else
        echo "WARNING: Using container as root user is allowed only for ADMINISTRATORS"
        echo "         if you did it by mistake please restart your session by command:"
        echo "         docker exec -it -u `id -un` CONTAINER COMMAND"
        echo "         or"
        echo "         docker exec -it -e HOST_USER_NAME=\`id -un\` -e HOST_USER_ID=\`id -u\` CONTAINER docker-entrypoint.sh [bash, YOUR COMMAND]"
        exec "$@"
    fi
fi
echo "docker-entrypoint exit"