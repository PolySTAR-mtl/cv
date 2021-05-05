gst-launch-1.0 nvcamerasrc fpsRange="60.0 60.0" ! 'video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)60/1' ! nvoverlaysink


