steps on how to build 


go to the file directory in terminal

1. run  docker buildx build --platform linux/amd64 -t  'the name tag that you want' .

2. run docker tag 'the name tag that you want' gcr.io/'project id'/'filename'

3. run docker push gcr.io/'project id'/'filename'

steps on to run in docker 

1. run  docker buildx build --platform linux/amd64 -t  'the name tag that you want' .

2. run docker run -p 'port to specify':8080 'the name tag that you want' , since it expose on 8080


to deploy 
1. go the container registry

2. go to conatiner

3. press cloud run 


5. set the port to 8080

if everything is done you can proceed to run in cloud

