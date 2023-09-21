if [ -d Build ]; then
    rm -rf Build/*
else
    mkdir Build
fi

cd Build && cmake .. && make && cd ..