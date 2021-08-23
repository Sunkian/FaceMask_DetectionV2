# FaceMask_DetectionV2

1. On the local terminal, start a video stream : ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0:none" -f rtsp -rtsp_transport tcp rtsp://10.60.16.156:8554/alicewebcam
1. OR from a local video file : ffmpeg -re -stream_loop -1 -i FaceMaskedTeam.mp4 -f rtsp -rtsp_transport tcp rtsp://10.60.16.156:8554/teamvideo
2. ssh andre@10.60.16.156
3. Go to apagnoux/FacemMask_DetectionV2
4. Type the following commands :
  -export INPUT_RTSP_URL=rtsp://10.60.16.156:8554/alicewebcam
  -export OUTPUT_RTSP_URL=rtsp://10.60.16.156:8554/facemaskalice
  -export MQTT_URL=10.60.16.156
  -export MQTT_TOPIC=topic/facemask_test
5. source ../env/bin/activate
6. python detect.ty
7. In another window but still on the remote server type : mosquitto_sub -t topic/facemask_test
