### Push a version tag

on master branch
git tag vx.x.x
git push origin vx.x.x


### Create release

Manually create a new release using the tag pushed in the step above.


### Building and pushing the Docker image

Step 1:
sudo docker build . --tag=lubo1994/pv-hawk:vx.x.x

Step 2:
sudo docker login

Step3:
sudo docker push lubo1994/pv-hawk:vx.x.x

Step 4:
sudo docker tag lubo1994/pv-hawk:v1.0.0 lubo1994/pv-hawk:latest

Step 5:
sudo docker push lubo1994/pv-hawk:latest


### Create a release for PV Hawk Viewer

After creating a new release of PV Hawk, create a release of PV Hawk Viewer with the same version number.
