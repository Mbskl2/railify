# Railfy 

Use this tool to digitize a Train Track Plan and output an SVG file marking 
properties of the edges. We also do output something that is closer to a mathematical Graph, a plot
for this can be found in the backend/temp files, after a run. 

Have Fun :)

## How it works
![Pipeline](https://github.com/Mbskl2/railify/blob/master/pipeline.png?raw=true)

### Sample input
![Input pdf](https://github.com/Mbskl2/railify/blob/master/input.png?raw=true)

### Sample output
![Output](https://github.com/Mbskl2/railify/blob/master/output.png?raw=true)

## Install

### Backend

```
pip install -r backend/requirements.txt
```

### Frontend

```
cd frontend
npm install 
```

## Run

### Start backend server

```
python backend/app.py
```

### Start backend server

```
cd frontend
npm run dev
```