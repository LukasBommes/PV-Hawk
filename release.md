### Push a version tag

on master branch
git tag vx.x.x
git push origin vx.x.x


### Create release

Manually create a new release using the tag pushed in the step above.


### Building and pushing the Docker image

Step 1:
sudo docker build . --tag=lubo1994/pv-drone-inspect:vx.x.x

Step 2:
sudo docker login

Step3:
sudo docker push lubo1994/pv-drone-inspect:vx.x.x

Step 4:
sudo docker tag lubo1994/pv-drone-inspect:v1.0.0 lubo1994/pv-drone-inspect:latest

Step 5:
sudo docker push lubo1994/pv-drone-inspect:latest


### Create a release for PV Drone Inspect Viewer

After creating a new release of PV Drone Inspect, create a release of PV Drone Inspect Viewer with the same version number.
